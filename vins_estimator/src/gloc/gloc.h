#pragma once

#include <atomic>
#include <ceres/ceres.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <limits>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <unordered_map>

#include "camodocal/camera_models/CameraFactory.h"
#include "colmap_util.h"
#include "dbow3_util.h"
#include "gloc_cost_functors.h"
#include "parameters.h"
#include "point_features.h"

// ── Gloc-local logging macros ─────────────────────────────────────────────────
//
// Bypass rosconsole entirely so gloc messages are always visible regardless
// of the active ROS log level. Filter with:
//
//   roslaunch ... 2>&1 | grep "^\[GLOC"
//
// Levels:
//   GLOC_DEBUG  verbose pipeline trace   stdout  [GLOC D]
//   GLOC_INFO   normal milestones        stdout  [GLOC I]
//   GLOC_WARN   unexpected but alive     stdout  [GLOC W]
//   GLOC_ERROR  unrecoverable failure    stderr  [GLOC E]
//
// All output is flushed immediately so messages appear in order with other
// terminal output even when stdout is line-buffered.
// ─────────────────────────────────────────────────────────────────────────────
// clang-format off
#define GLOC_DEBUG(fmt, ...) \
    do { std::fprintf(stdout, "[GLOC D] " fmt "\n", ##__VA_ARGS__); \
         std::fflush(stdout); } while (0)
#define GLOC_INFO(fmt, ...) \
    do { std::fprintf(stdout, "[GLOC I] " fmt "\n", ##__VA_ARGS__); \
         std::fflush(stdout); } while (0)
#define GLOC_WARN(fmt, ...) \
    do { std::fprintf(stdout, "[GLOC W] " fmt "\n", ##__VA_ARGS__); \
         std::fflush(stdout); } while (0)
#define GLOC_ERROR(fmt, ...) \
    do { std::fprintf(stderr, "[GLOC E] " fmt "\n", ##__VA_ARGS__); \
         std::fflush(stderr); } while (0)
// clang-format on

// Forward-declare Estimator so gloc.h does not pull in all estimator headers.
namespace vins_multi
{
class Estimator;
};

namespace gloc
{

// ─────────────────────────────────────────────────────────────────────────────
// CameraRingBuffer
//
// Per-module ring of (timestamp, image) pairs with eviction protection.
//
// Producer:  rosNode callback for that module (single producer per buffer).
// Consumer:  Gloc process thread (single consumer per buffer).
//
// Two eviction mechanisms keep the buffer bounded:
//
//   1. trimOlderThan(t)  — called by Phase 3 of onSnapshotChanged after
//      computing the oldest unresolved keyframe. This is the *informed*
//      eviction: it only removes images that no current keyframe can need.
//
//   2. Capacity eviction in push() — a safety ceiling that prevents
//      unbounded growth if onSnapshotChanged stops running. push() honours
//      a protect floor: images at or after protect_floor_ are kept even
//      when the soft capacity is exceeded, up to a hard ceiling of
//      hard_capacity_ (= 3× soft cap by default). This ensures that images
//      gloc still needs survive bursts of incoming frames during VINS stalls.
//
// The eviction watermark (t_evict_watermark_) tracks the highest timestamp
// lost to capacity eviction. findNearest uses it to distinguish a real gap
// in the camera stream (→ NoneInTolerance) from an image that existed but
// was evicted before gloc could look (→ Evicted, treated as NoneInTolerance
// with a warning).
//
// All entries are kept in ascending timestamp order: push appends at the
// back, eviction removes from the front. ROS image callbacks deliver frames
// in arrival order, which is normally monotonic in timestamp, so the
// invariant is preserved by construction. The push method drops out-of-order
// frames (rare; only happens with severely delayed or replayed bags).
// ─────────────────────────────────────────────────────────────────────────────

enum class LookupVerdict
{
    // A frame within tolerance of t_kf was found.
    Found,

    // No frame within tolerance, but the buffer's newest frame is older than
    // t_kf + tol — relevant frames may still arrive. The caller should retry
    // on the next round.
    NotYet,

    // No frame within tolerance, and the buffer has moved past t_kf + tol.
    // The gap is genuine (the camera stream had no image near t_kf).
    // No future arrival can satisfy this keyframe. Caller should commit.
    NoneInTolerance,

    // No frame within tolerance, and the buffer has moved past t_kf + tol,
    // BUT the eviction watermark indicates that an image in the tolerance
    // window was lost to capacity eviction before gloc could look. This is
    // treated as terminal (the image is gone), but logged as a warning so
    // the operator can increase the ring buffer capacity.
    Evicted,

    // Buffer is empty.
    Empty,
};

struct LookupResult
{
    LookupVerdict verdict{LookupVerdict::Empty};
    double t_image{0.0}; // timestamp of the returned image, valid iff Found
    cv::Mat image;       // valid iff Found
};

class CameraRingBuffer
{
  public:
    explicit CameraRingBuffer(std::size_t capacity = 10)
        : capacity_{capacity},
          hard_capacity_{capacity * 3}
    {
    }

    // Append an image with timestamp t. Drops out-of-order frames silently.
    // Evicts images below protect_floor_ when soft capacity is exceeded.
    // Above soft capacity, protected images are kept until the hard ceiling.
    void push(double t, const cv::Mat &image);

    // Find the frame with timestamp nearest to t_kf within ±tol. See
    // LookupResult / LookupVerdict for the contract.
    LookupResult findNearest(double t_kf, double tol) const;

    // Drop all slots whose timestamp is strictly less than t. The slot at
    // exactly t (if any) is kept.
    void trimOlderThan(double t);

    // Set the protect floor: push() will not evict images with t >= floor
    // unless the hard ceiling is reached. Called by onSnapshotChanged Phase 3
    // after computing the oldest-unresolved time minus tolerance. Pass
    // std::numeric_limits<double>::max() to disable protection (all slots
    // resolved) — this ensures no timestamp compares >= the floor.
    void setProtectFloor(double floor);

    // Diagnostic helpers.
    std::size_t size() const;
    bool empty() const;

  private:
    struct Slot
    {
        double t;
        cv::Mat image; // cv::Mat is refcounted; storing by value is cheap
    };

    mutable std::mutex mtx_;
    std::deque<Slot> slots_;
    std::size_t capacity_;      // soft capacity (configured via YAML)
    std::size_t hard_capacity_; // absolute ceiling (3× soft)

    // Images at or after this timestamp are protected from capacity eviction.
    // Updated by setProtectFloor() from onSnapshotChanged.
    //
    // Sentinel: std::numeric_limits<double>::max() means "no protection" —
    // no image timestamp can be >= max(), so push() evicts freely at the
    // soft capacity. DO NOT use 0.0 as "no protection": ROS timestamps are
    // large positive numbers, so 0.0 would protect everything.
    double protect_floor_{std::numeric_limits<double>::max()};

    // Highest timestamp evicted by capacity pressure (not by trimOlderThan,
    // which is an informed eviction). Used by findNearest to detect whether
    // a NoneInTolerance was caused by a genuine stream gap or by a lost image.
    double t_evict_watermark_{0.0};
};

// ─────────────────────────────────────────────────────────────────────────────
// Map
//
// All data loaded during init() that describes the reference map.
// Populated by loadColmapData() and loadDatabase().
// ─────────────────────────────────────────────────────────────────────────────

struct Map
{
    // ── COLMAP ───────────────────────────────────────────────────────────────

    // All map images, sorted by image_id, poses in world frame.
    std::vector<colmap::Image> images;

    // camera_id → calibration
    colmap::CalibMap calibs;

    // camera_id → rig extrinsic (R_cam_rig, t_cam_rig)
    std::map<uint32_t, colmap::CamRigTransform> cam_rig_map;

    // ── DBoW3 ────────────────────────────────────────────────────────────────

    DBoW3::Database db;

    // Per-image features, index-aligned with images
    std::vector<dbow3::ImageFeatures> feats;
};

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot
//
// What the estimator hands to gloc when the keyframe window changes.
//
// Plain-data: copies of the keyframe times + local poses, ordered by time.
// No pointers into estimator internals — gloc owns this copy.
//
// The snapshot is the WHOLE current window, not just newly-added keyframes.
// Gloc figures out which keyframes are new by comparing against its own
// state_map_; carried-over keyframes get their local pose refreshed.
// ─────────────────────────────────────────────────────────────────────────────

struct Snapshot
{
    struct KeyframeEntry
    {
        // Identity & lookup key for gloc. Real-time timestamp, post time-offset.
        double t_kf{0.0};

        // Raw image timestamp WITHOUT td correction. This is the timestamp
        // domain that the gloc ring buffers operate in (images are pushed at
        // their raw ROS header stamp). All ring-buffer lookups use t_image
        // so that the dynamically-estimated td does not create a domain
        // mismatch between keyframe queries and buffered images.
        double t_image{0.0};

        // Local-frame body pose at t_kf: T_local_body(t_kf).
        // Gloc holds this as the soft-constraint anchor for the corresponding
        // optimization variable T_map_body(t_kf).
        Eigen::Quaterniond R_local{Eigen::Quaterniond::Identity()};
        Eigen::Vector3d P_local{Eigen::Vector3d::Zero()};

        // VINS camera module that captured this keyframe (index into
        // CAM_MODULES). Informational only — gloc queries every gloc module's
        // ring buffer regardless of which VINS camera captured the keyframe.
        unsigned int cam_unique_id{0};
    };

    std::vector<KeyframeEntry> keyframes; // ascending by t_kf
    uint64_t snapshot_id{0};              // monotonic, for logging
};

// ─────────────────────────────────────────────────────────────────────────────
// PerModuleResolution
//
// One slot per (keyframe, gloc-module) pair. Each slot is independently
// in state-1 (Found, image cached), state-2 (NoneInTolerance, definitive no),
// or state-3 (Empty / NotYet, will retry next snapshot).
//
// Heavier per-slot artefacts (ORB features, DBoW3 match, 2D-2D
// correspondences) will be cached here too once that pipeline is wired up.
// Initially we only store the image.
// ─────────────────────────────────────────────────────────────────────────────

struct PerModuleResolution
{
    LookupVerdict verdict{LookupVerdict::Empty};

    // Valid when verdict == Found.
    double t_image{0.0};
    cv::Mat image;              // raw image from ring buffer
    cv::Mat preprocessed_image; // preprocessed version (filled by runOrbAndDbow)

    // ── Pipeline cache (filled by processLoop, stage 1) ───────────────────

    // True once ORB extraction + DBoW query + correspondence have been run
    // for this slot. Guards against reprocessing on subsequent rounds.
    bool pipeline_done{false};

    // ORB features extracted from `image`.
    dbow3::ImageFeatures query_feats;

    // DBoW3 top-N candidates: (score, train_image_index into map_.images).
    // Populated during ORB/DBoW stage; entries are sorted descending by score.
    std::vector<std::pair<double, size_t>> dbow_candidates;

    // Index into map_.images of the winning train image after consensus
    // voting.  -1 means this slot was rejected by the vote filter.
    int best_train_idx{-1};

    // All accepted train image indices after consensus voting, sorted
    // descending by DBoW3 score. best_train_idx == voted_train_idxs[0]
    // when non-empty. Size capped at GLOC_VOTE_MAX_MATCHES.
    std::vector<int> voted_train_idxs;

    // 2D-2D point correspondences between this query image and the winning
    // train image.  In original (distorted) pixel coordinates.
    std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> pt_pairs_distorted;

    // Same correspondences with lens distortion removed.
    std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> pt_pairs_undistorted;

    // Per-match correspondences for all voted_train_idxs entries.
    // multi_pt_pairs_distorted[k]   ↔ voted_train_idxs[k]
    // multi_pt_pairs_undistorted[k] ↔ voted_train_idxs[k]
    // Index 0 duplicates pt_pairs_* for backward compatibility.
    std::vector<std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>> multi_pt_pairs_distorted;
    std::vector<std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>> multi_pt_pairs_undistorted;

    bool isTerminal() const
    {
        return verdict == LookupVerdict::Found ||
               verdict == LookupVerdict::NoneInTolerance ||
               verdict == LookupVerdict::Evicted;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// KeyframeGlocState
//
// Per-keyframe state carried across snapshots. Lives in Gloc::state_map_
// keyed by t_kf. The local pose is refreshed every snapshot; the per-module
// resolutions are filled in once per slot and never recomputed.
// ─────────────────────────────────────────────────────────────────────────────

struct KeyframeGlocState
{
    double t_kf{0.0};    // == map key (td-corrected, used for pose optimization)
    double t_image{0.0}; // raw image timestamp (no td), used for ring-buffer lookup
    Eigen::Quaterniond R_local{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d P_local{Eigen::Vector3d::Zero()};
    unsigned int cam_unique_id{0};

    // One slot per gloc module, sized to GLOC_CAM_MODULES.size() at insertion.
    std::vector<PerModuleResolution> per_gloc;
};

class Gloc
{
  public:
    Gloc();
    ~Gloc();

    // ─────────────────────────────────────────────────────────────────────────
    // setEstimator
    //
    // Must be called before init().
    // Stores a non-owning pointer to the estimator so the process loop can
    // read VIO output from it.
    // ─────────────────────────────────────────────────────────────────────────
    void setEstimator(vins_multi::Estimator *estimator);

    // ─────────────────────────────────────────────────────────────────────────
    // init
    //
    // Call once after setEstimator() and before start_process_thread().
    // Reads all COLMAP data and loads the DBoW3 database using the paths
    // from the gloc:: parameters (set by readParameters()).
    //
    // Allocates one CameraRingBuffer per entry in GLOC_CAM_MODULES, so the
    // gloc-side unique_id (the index into GLOC_CAM_MODULES) maps directly to
    // the index into ring_buffers_.
    //
    // Checks that all required paths exist before attempting to load anything.
    //
    // @returns  true  on success — ready to localise.
    //           false on any missing path, file error, or inconsistent data.
    // ─────────────────────────────────────────────────────────────────────────
    bool init();

    // Returns the loaded COLMAP map. Valid after init() succeeds.
    const Map &getMap() const
    {
        return map_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // setTMapLocalCallback
    //
    // Register a callback fired on the gloc worker thread whenever a new
    // T_map_local is accepted by runOptimization.
    //
    // Convention:  X_world = R * X_local + t
    //
    // The callback must be lightweight (e.g. store under a mutex and return).
    // It runs on the gloc worker thread — do NOT acquire any lock that the
    // caller's thread also holds, to avoid deadlock.
    // ─────────────────────────────────────────────────────────────────────────
    using TMapLocalCallback = std::function<void(const Eigen::Matrix3d &R,
                                                 const Eigen::Vector3d &t)>;
    void setTMapLocalCallback(TMapLocalCallback cb);

    // ─────────────────────────────────────────────────────────────────────────
    // start_process_thread
    //
    // Starts the background processing loop.
    // Call after init() succeeds.
    // The loop runs until the destructor signals it to stop.
    // ─────────────────────────────────────────────────────────────────────────
    void start_process_thread();

    // ─────────────────────────────────────────────────────────────────────────
    // pushCameraImage
    //
    // Append an image to the ring buffer for the given gloc-side module
    // (gloc_unique_id is the index into GLOC_CAM_MODULES). Called from the
    // rosNode image callback. Non-blocking.
    //
    // No-op when init() has not yet succeeded — safe to call before / outside
    // gloc being enabled. Guards against out-of-bounds gloc_unique_id.
    // ─────────────────────────────────────────────────────────────────────────
    void pushCameraImage(unsigned int gloc_unique_id,
                         double t,
                         const cv::Mat &image);

    // ─────────────────────────────────────────────────────────────────────────
    // findNearestImage
    //
    // Look up the nearest image in the ring buffer for a given gloc-side
    // module. Used by the process loop to resolve a keyframe's gloc image
    // exactly once (state-3 → state-1/2 transition).
    //
    // Returns a LookupResult; an out-of-range gloc_unique_id or an
    // uninitialized Gloc returns {Empty}.
    // ─────────────────────────────────────────────────────────────────────────
    LookupResult findNearestImage(unsigned int gloc_unique_id,
                                  double t_kf,
                                  double tol) const;

    // Number of gloc-side modules (== GLOC_CAM_MODULES.size()), or 0 if not
    // initialised. Useful for rosNode to size its parallel-callback array.
    std::size_t numGlocModules() const
    {
        return ring_buffers_.size();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // onSnapshotChanged
    //
    // Called by the estimator thread after each keyframe-window change
    // (typically end of Estimator::processWindow). Synchronous, returns in
    // ~μs once the eager-resolution path is hot.
    //
    // Three things happen under state_mutex_:
    //
    //   1. Reconcile state_map_ with the new snapshot:
    //        * for each keyframe in snapshot: insert if new, refresh local
    //          pose if existing.
    //        * remove map entries whose t_kf is no longer in the snapshot
    //          (marginalised out of the VINS window).
    //
    //   2. Eager per-slot resolution: for every (keyframe, gloc-module) slot
    //      that is not yet terminal (i.e. verdict is Empty or NotYet), call
    //      ring_buffers_[g]->findNearest(t_kf, tol_g). Update the slot's
    //      verdict. On Found: cache the image and call ring_buffers_[g]
    //      ->trimOlderThan(t_image) to free slots that can never serve a
    //      future (later) keyframe.
    //
    //   3. Signal the worker thread that a fresh snapshot exists.
    //
    // The Ceres solve runs on the worker thread without holding state_mutex_,
    // so this call only contends with the worker for brief snapshot-copy ops.
    // The estimator is not stalled by an in-progress optimisation.
    //
    // No-op when gloc is not initialised.
    // ─────────────────────────────────────────────────────────────────────────
    void onSnapshotChanged(const Snapshot &snapshot);

  private:
    // ── Helpers called by init() ─────────────────────────────────────────────

    // Verify that all paths required for init() exist on disk.
    // Prints a specific error for each missing path.
    // Returns false if anything is missing.
    bool checkPaths() const;

    // Read images.bin, cameras.bin, rigs.bin, world_transform.txt.
    // Populates map_.images, map_.calibs, map_.cam_rig_map.
    // Returns false on any error.
    bool loadColmapData();

    // Load vocabulary + feature cache + DBoW3 database.
    // Populates map_.db and map_.feats.
    // Returns false on any error.
    bool loadDatabase();

    // ── ORB extractor and feature matcher (constructed once in init()) ───────
    //
    // Shared across all processLoop rounds. Both are used only on the worker
    // thread so no locking is needed.
    std::unique_ptr<PointFeatureExtractor> orb_extractor_;
    cv::Ptr<cv::DescriptorExtractor> beblid_extractor_; // null if GLOC_USE_BEBLID==0
    std::unique_ptr<PointFeatureMatcher> feat_matcher_;

    // One camera model per gloc module, loaded from GLOC_CAM_MODULES[g].calib_file_[0].
    // Used in runCorrespondences to undistort query keypoints.
    std::vector<camodocal::CameraPtr> query_cameras_;

    // ── Process loop ─────────────────────────────────────────────────────────

    // Entry point for process_thread_.
    void processLoop();

    // ── T_map_local callback ──────────────────────────────────────────────────
    mutable std::mutex cb_mutex_;
    TMapLocalCallback callback_;

    // ── Snapped state + snap poses ────────────────────────────────────────────
    // Protected by snap_mutex_.
    // Written by runOptimization (worker thread).
    // Read by onSnapshotChanged (estimator thread) and init (main thread).
    mutable std::mutex snap_mutex_;
    //
    // snapped_ becomes true after the first successful optimization round.
    // Once snapped, T_map_local_R_ and T_map_local_t_ hold the last accepted
    // estimate of T_map_local (world ← local) and are used as a prior in
    // subsequent rounds.
    //
    // T_map_local convention:
    //   X_world = T_map_local_R_ * X_local + T_map_local_t_
    bool snapped_{false};
    Eigen::Matrix3d T_map_local_R_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d T_map_local_t_{Eigen::Vector3d::Zero()};

    // Snapshot of local poses used in the last successful solve.
    // Keyed by t_kf so we can match against new snapshots by timestamp.
    // Used to compute ΔT_map_local when VINS re-adjusts local poses.
    struct SnapPose
    {
        Eigen::Quaterniond R_local; // R_local_body at solve time
        Eigen::Vector3d P_local;    // P_local_body at solve time
        double weight;              // correspondence count (for weighted mean)
    };
    std::map<double, SnapPose> last_snap_poses_; // t_kf → SnapPose

    // Stage 2: build and solve the Ceres problem on working_set.
    // Returns true if the solution was accepted (inlier ratio ≥ threshold)
    // and updates snapped_ / T_map_local_R_ / T_map_local_t_ on success.
    bool runOptimization(std::vector<KeyframeGlocState> &working_set);
    bool runOptimization_6DOF(std::vector<KeyframeGlocState> &working_set);
    bool runOptimization_4DOF(std::vector<KeyframeGlocState> &working_set);
    bool runOptimization_FixedRel_6DOF(std::vector<KeyframeGlocState> &working_set);
    bool runOptimization_FixedRel_4DOF(std::vector<KeyframeGlocState> &working_set);
    bool runOptimization_FixedRel(std::vector<KeyframeGlocState> &working_set, bool use_4dof);
    // features from the cached image and query DBoW3 to populate dbow_candidates.
    void runOrbAndDbow(std::vector<KeyframeGlocState> &working_set);

    // Stage 1b: cross-keyframe magnitude-consistency voting. For each candidate
    // (i,n), count how many other keyframes have a candidate whose world-distance
    // matches the local-pose distance. Sets best_train_idx per slot.
    void runConsensusVoting(std::vector<KeyframeGlocState> &working_set);

    // Stage 1c: for each slot with a valid best_train_idx, match ORB descriptors
    // against the cached train features, run geometric verification, and store
    // the surviving 2D-2D correspondences.
    void runCorrespondences(std::vector<KeyframeGlocState> &working_set,
                            uint64_t snapshot_id);

    // Stage 1d: re-acquire state_mutex_ briefly and flush pipeline results from
    // the working_set copy back into state_map_.
    void writeBackToStateMap(const std::vector<KeyframeGlocState> &working_set);

    // ── Estimator (non-owning) ───────────────────────────────────────────────

    // Set by setEstimator(). Used in processLoop() to read VIO state.
    vins_multi::Estimator *estimator_ptr_{nullptr};

    // ── Map data ───────────────────────────────────────────────────────

    Map map_;

    // ── Per-module image ring buffers ────────────────────────────────────────
    //
    // Allocated in init(); indexed by gloc-side unique_id (= index into
    // GLOC_CAM_MODULES). Empty until init() succeeds, so pushCameraImage
    // safely no-ops when gloc is disabled or not yet initialised.
    //
    // unique_ptr is used so the vector can be sized without moving non-movable
    // CameraRingBuffer instances (each holds a std::mutex).
    std::vector<std::unique_ptr<CameraRingBuffer>> ring_buffers_;

    // True once init() has populated ring_buffers_ and the map. pushCameraImage
    // and findNearestImage check this before doing any work.
    std::atomic<bool> initialized_{false};

    // ── Snapshot / per-keyframe state ────────────────────────────────────────
    //
    // state_map_ holds the union of all keyframes that have appeared in any
    // snapshot since gloc started, minus those marginalised out of the VINS
    // window. Keyed by t_kf so insertion-by-key is idempotent.
    //
    // Guarded by state_mutex_, which is also the cv mutex for work_cv_. The
    // worker thread takes this mutex briefly to copy out a working set, then
    // runs the Ceres solve without holding it.
    //
    // snapshot_fresh_ is set true by onSnapshotChanged after a successful
    // update and cleared by the worker when it picks up the work. Coalescing:
    // multiple notifications during a long solve collapse into a single
    // "snapshot is fresh, optimize again" signal.
    //
    // last_snapshot_id_ tracks the most recent snapshot_id observed by
    // onSnapshotChanged, mostly for logging.
    mutable std::mutex state_mutex_;
    std::map<double, KeyframeGlocState> state_map_;
    bool snapshot_fresh_{false};
    uint64_t last_snapshot_id_{0};
    // Sensor timestamp of the first snapshot received. Used to enforce
    // a startup delay before processing begins.
    double first_snapshot_time_{-1.0};

    // ── Thread state ─────────────────────────────────────────────────────────

    std::thread process_thread_;
    std::condition_variable work_cv_; // signalled on snapshot_fresh_ change
    bool stop_thread_{false};
};

} // namespace gloc