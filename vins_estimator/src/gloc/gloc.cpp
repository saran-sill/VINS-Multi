#include "gloc.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>

#include "estimator/parameters.h" // for vins_multi::GLOC_CAM_MODULES

namespace fs = std::filesystem;

namespace gloc
{

// ─────────────────────────────────────────────────────────────────────────────
// CameraRingBuffer
// ─────────────────────────────────────────────────────────────────────────────

void CameraRingBuffer::push(double t, const cv::Mat &image)
{
    std::lock_guard<std::mutex> lk(mtx_);

    // Reject out-of-order frames. The invariant slots_ is sorted ascending
    // by timestamp keeps findNearest correct without sorting on every read.
    // Normal ROS callbacks deliver frames in arrival order, which is almost
    // always monotonic in timestamp. Stragglers from a delayed transport or
    // a replayed bag are dropped rather than inserted out of order.
    if (!slots_.empty() && t <= slots_.back().t)
    {
        ROS_WARN_THROTTLE(2.0,
                          "[Gloc::CameraRingBuffer] dropping out-of-order frame: "
                          "t=%.6f vs newest=%.6f",
                          t, slots_.back().t);
        return;
    }

    slots_.push_back({t, image});

    while (slots_.size() > capacity_)
        slots_.pop_front();
}

LookupResult CameraRingBuffer::findNearest(double t_kf, double tol) const
{
    std::lock_guard<std::mutex> lk(mtx_);

    LookupResult r;

    if (slots_.empty())
    {
        r.verdict = LookupVerdict::Empty;
        return r;
    }

    // slots_ is sorted ascending by timestamp. Binary-search the first slot
    // with t >= t_kf, then compare its predecessor too — the nearest in time
    // is one of those two.
    auto it = std::lower_bound(
        slots_.begin(), slots_.end(), t_kf,
        [](const Slot &s, double t) { return s.t < t; });

    auto best_it = slots_.end();
    double best_dt = std::numeric_limits<double>::infinity();

    if (it != slots_.end())
    {
        const double dt = std::abs(it->t - t_kf);
        if (dt < best_dt)
        {
            best_dt = dt;
            best_it = it;
        }
    }
    if (it != slots_.begin())
    {
        auto prev_it = std::prev(it);
        const double dt = std::abs(prev_it->t - t_kf);
        if (dt < best_dt)
        {
            best_dt = dt;
            best_it = prev_it;
        }
    }

    if (best_it != slots_.end() && best_dt <= tol)
    {
        r.verdict = LookupVerdict::Found;
        r.t_image = best_it->t;
        r.image = best_it->image; // shallow refcount-bumped copy
        return r;
    }

    // No frame within tolerance. Distinguish "still waiting" from "missed it
    // forever" by checking whether the buffer has any frame beyond t_kf+tol.
    // If the newest frame is still earlier than t_kf+tol, a future arrival
    // may yet fall inside the window, so we tell the caller to retry.
    const double newest_t = slots_.back().t;
    r.verdict = (newest_t > t_kf + tol)
                    ? LookupVerdict::NoneInTolerance
                    : LookupVerdict::NotYet;
    return r;
}

void CameraRingBuffer::trimOlderThan(double t)
{
    std::lock_guard<std::mutex> lk(mtx_);

    // Strict less-than: the slot at exactly t (if present) is kept. This is
    // the "drop older-than-matched" optimisation called by Gloc after a
    // successful findNearest — future state-3 keyframes will have t_kf later
    // than the just-resolved one, so anything earlier than the matched slot
    // can never serve them.
    auto cutoff = std::lower_bound(
        slots_.begin(), slots_.end(), t,
        [](const Slot &s, double v) { return s.t < v; });

    slots_.erase(slots_.begin(), cutoff);
}

std::size_t CameraRingBuffer::size() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return slots_.size();
}

bool CameraRingBuffer::empty() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return slots_.empty();
}

Gloc::Gloc()
{
}

Gloc::~Gloc()
{
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        stop_thread_ = true;
    }
    work_cv_.notify_all();

    if (process_thread_.joinable())
        process_thread_.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// setEstimator
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::setEstimator(vins_multi::Estimator *estimator)
{
    estimator_ptr_ = estimator;
}

// ─────────────────────────────────────────────────────────────────────────────
// init
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::init()
{
    if (estimator_ptr_ == nullptr)
    {
        std::cerr << "[Gloc] ERROR: setEstimator() must be called before init().\n";
        return false;
    }

    std::cout << "[Gloc::init] Initialising global localiser ...\n";

    if (!checkPaths())
        return false;

    if (!loadColmapData())
        return false;

    if (!loadDatabase())
        return false;

    // Allocate one ring buffer per gloc-side camera module. The gloc-side
    // unique_id is the index into GLOC_CAM_MODULES, which we mirror here so
    // pushCameraImage / findNearestImage can index directly. unique_ptr
    // wrapper is needed because CameraRingBuffer holds a non-movable mutex.
    //
    // Capacity is per-module, read from image_ring_buffer_capacity_ on the
    // gloc-side camera_module_info (defaults to 10; overridable per-cam via
    // the image_ring_buffer_capacity field in YAML). A slow gloc camera can
    // use a smaller buffer than a fast one; both can be tuned independently
    // alongside image_match_tol_s_.
    ring_buffers_.clear();
    ring_buffers_.reserve(vins_multi::GLOC_CAM_MODULES.size());
    for (std::size_t i = 0; i < vins_multi::GLOC_CAM_MODULES.size(); ++i)
    {
        const std::size_t cap = static_cast<std::size_t>(
            vins_multi::GLOC_CAM_MODULES[i].image_ring_buffer_capacity_);
        ring_buffers_.emplace_back(std::make_unique<CameraRingBuffer>(cap));
    }

    std::cout << "[Gloc::init] Allocated " << ring_buffers_.size()
              << " camera ring buffer(s).\n";

    initialized_.store(true, std::memory_order_release);

    std::cout << "[Gloc::init] Ready.\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// start_process_thread
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::start_process_thread()
{
    process_thread_ = std::thread(&Gloc::processLoop, this);
    std::cout << "[Gloc] Process thread started.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// pushCameraImage
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::pushCameraImage(unsigned int gloc_unique_id,
                           double t,
                           const cv::Mat &image)
{
    // Fast no-op when gloc is disabled or init() has not yet completed.
    // The callback can call this unconditionally; checks live here.
    if (!initialized_.load(std::memory_order_acquire))
        return;

    if (gloc_unique_id >= ring_buffers_.size())
    {
        ROS_WARN_THROTTLE(2.0,
                          "[Gloc::pushCameraImage] gloc_unique_id %u out of range (size=%zu)",
                          gloc_unique_id, ring_buffers_.size());
        return;
    }

    ring_buffers_[gloc_unique_id]->push(t, image);
}

// ─────────────────────────────────────────────────────────────────────────────
// findNearestImage
// ─────────────────────────────────────────────────────────────────────────────

LookupResult Gloc::findNearestImage(unsigned int gloc_unique_id,
                                    double t_kf,
                                    double tol) const
{
    if (!initialized_.load(std::memory_order_acquire))
        return {LookupVerdict::Empty, 0.0, cv::Mat{}};

    if (gloc_unique_id >= ring_buffers_.size())
        return {LookupVerdict::Empty, 0.0, cv::Mat{}};

    return ring_buffers_[gloc_unique_id]->findNearest(t_kf, tol);
}

// ─────────────────────────────────────────────────────────────────────────────
// onSnapshotChanged
//
// Called synchronously by the estimator after each keyframe-window change.
// Three phases under state_mutex_:
//   1. reconcile state_map_ with the snapshot
//   2. eager per-slot resolution against the ring buffers
//   3. signal the worker
// The Ceres solve runs on the worker thread without holding state_mutex_,
// so phase-1/2 contention is bounded to μs.
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::onSnapshotChanged(const Snapshot &snapshot)
{
    if (!initialized_.load(std::memory_order_acquire))
        return;

    const std::size_t n_modules = ring_buffers_.size();

    {
        std::lock_guard<std::mutex> lk(state_mutex_);

        last_snapshot_id_ = snapshot.snapshot_id;

        // ── Phase 1: reconcile state_map_ with snapshot ─────────────────────
        //
        // Build a set of snapshot t_kf values for fast "is this key still
        // present?" lookup. snapshot.keyframes is ordered ascending, so a
        // single linear merge would also work; std::set is clearer.
        std::set<double> snapshot_keys;
        for (const auto &kf : snapshot.keyframes)
            snapshot_keys.insert(kf.t_kf);

        // Drop entries whose t_kf is no longer in the window. These keyframes
        // have been marginalised out by VINS; their state (image, future
        // ORB/DBoW caches) is no longer needed.
        for (auto it = state_map_.begin(); it != state_map_.end();)
        {
            if (snapshot_keys.find(it->first) == snapshot_keys.end())
                it = state_map_.erase(it);
            else
                ++it;
        }

        // Insert new entries; refresh local pose on existing ones. The
        // per_gloc vector is sized to n_modules on insertion and never
        // resized again (number of gloc modules is fixed at init).
        for (const auto &kf : snapshot.keyframes)
        {
            auto [it, inserted] = state_map_.try_emplace(kf.t_kf);
            auto &s = it->second;
            if (inserted)
            {
                s.t_kf = kf.t_kf;
                s.cam_unique_id = kf.cam_unique_id;
                s.per_gloc.assign(n_modules, PerModuleResolution{});
            }
            // Refresh local pose every snapshot — VINS keeps re-optimising
            // these and the soft-constraint anchor needs to track.
            s.R_local = kf.R_local;
            s.P_local = kf.P_local;
        }

        // ── Phase 2: eager per-slot resolution ──────────────────────────────
        //
        // For each (keyframe, gloc-module) slot that is still non-terminal
        // (Empty or NotYet), attempt to resolve it now against the module's
        // ring buffer. On a successful Found we additionally trim the buffer
        // to drop slots older than the matched one — they can never serve a
        // future keyframe under the t_kf-increasing invariant.
        for (auto &[t_kf, s] : state_map_)
        {
            for (std::size_t g = 0; g < s.per_gloc.size(); ++g)
            {
                auto &slot = s.per_gloc[g];
                if (slot.isTerminal())
                    continue; // Found or NoneInTolerance — already done

                const double tol_g = vins_multi::GLOC_CAM_MODULES[g].image_match_tol_s_;
                const LookupResult r = ring_buffers_[g]->findNearest(t_kf, tol_g);

                slot.verdict = r.verdict;
                if (r.verdict == LookupVerdict::Found)
                {
                    slot.t_image = r.t_image;
                    slot.image = r.image;
                    // Drop older-than-matched: future state-3 keyframes have
                    // larger t_kf, so anything earlier than r.t_image cannot
                    // help them. The matched slot itself is kept (strict <).
                    ring_buffers_[g]->trimOlderThan(r.t_image);
                }
                // For NotYet / NoneInTolerance / Empty we just record the
                // verdict; nothing else changes. NotYet → tried again next
                // snapshot; NoneInTolerance → terminal, never retried.
            }
        }

        // ── Phase 3: signal the worker ──────────────────────────────────────
        snapshot_fresh_ = true;
    }

    work_cv_.notify_one();
}

// ─────────────────────────────────────────────────────────────────────────────
// processLoop
//
// Single worker thread. Sleeps on work_cv_ until onSnapshotChanged signals
// that the state_map_ has fresh content. Then takes a working copy of the
// resolved (Found) per-keyframe state under state_mutex_, releases the lock,
// and runs the (currently TODO) heavy pipeline:
//   - for each new Found slot: ORB extract, DBoW3 query, 2D-2D correspondence
//   - assemble per-keyframe map-pose variables and gloc factors
//   - run Ceres solve
//   - collapse per-keyframe map poses into a single T_map_local
//   - publish T_map_local (atomic swap)
//
// Coalescing: if onSnapshotChanged fires N times during one Ceres solve,
// only one extra round is queued (snapshot_fresh_ stays true). The worker
// always picks up the newest state when it wakes.
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::processLoop()
{
    std::cout << "[Gloc] processLoop running.\n";

    while (true)
    {
        std::vector<KeyframeGlocState> working_set;
        uint64_t working_snapshot_id = 0;

        {
            std::unique_lock<std::mutex> lk(state_mutex_);
            work_cv_.wait(lk, [this] { return stop_thread_ || snapshot_fresh_; });
            if (stop_thread_)
                break;

            snapshot_fresh_ = false;
            working_snapshot_id = last_snapshot_id_;

            // Copy out only keyframes with at least one Found slot — keyframes
            // with zero resolved gloc images contribute nothing to the
            // current round but stay in state_map_ in case they resolve later.
            for (const auto &[t, s] : state_map_)
            {
                bool any_found = false;
                for (const auto &slot : s.per_gloc)
                {
                    if (slot.verdict == LookupVerdict::Found)
                    {
                        any_found = true;
                        break;
                    }
                }
                if (any_found)
                    working_set.push_back(s);
            }
        }
        // state_mutex_ released — the Ceres solve below runs without it.

        ROS_DEBUG("[Gloc] worker round snapshot_id=%lu, %zu resolved keyframes",
                  (unsigned long)working_snapshot_id, working_set.size());

        // TODO (next stage):
        //   1. For each Found slot in working_set whose ORB/DBoW/correspondences
        //      have not yet been computed, run that pipeline now and cache the
        //      results back on state_map_ (under a brief state_mutex_ acquire).
        //   2. Build the Ceres problem: per-keyframe T_map_body variables,
        //      soft-constraint factors between adjacent keyframes, gloc factors
        //      from cached correspondences.
        //   3. Solve.
        //   4. Collapse optimised T_map_body's into a weighted SE(3) mean
        //      T_map_local.
        //   5. Atomically publish T_map_local for consumers.
    }

    std::cout << "[Gloc] processLoop exiting.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// checkPaths
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::checkPaths() const
{
    bool ok = true;

    const std::vector<std::string> colmap_files = {
        "images.bin",
        "cameras.bin",
        "rigs.bin",
    };
    for (const auto &f : colmap_files)
    {
        const fs::path p = fs::path(GLOC_COLMAP_FOLDER) / f;
        if (!fs::exists(p))
        {
            std::cerr << "[Gloc] ERROR: missing COLMAP file: " << p << "\n";
            ok = false;
        }
    }

    if (!fs::exists(GLOC_COLMAP_IMG_FOLDER))
    {
        std::cerr << "[Gloc] ERROR: image folder does not exist: "
                  << GLOC_COLMAP_IMG_FOLDER << "\n";
        ok = false;
    }

    if (!fs::exists(GLOC_DBOW3_DATABASE))
    {
        std::cerr << "[Gloc] ERROR: DBoW3 database not found: "
                  << GLOC_DBOW3_DATABASE << "\n";
        ok = false;
    }

    if (!fs::exists(GLOC_DBOW3_VOCAB))
    {
        std::cerr << "[Gloc] ERROR: DBoW3 vocabulary not found: "
                  << GLOC_DBOW3_VOCAB << "\n";
        ok = false;
    }

    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadColmapData
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::loadColmapData()
{
    const std::string &colmap_dir = GLOC_COLMAP_FOLDER;
    const std::string &img_folder = GLOC_COLMAP_IMG_FOLDER;

    std::cout << "[Gloc] Loading COLMAP data from: " << colmap_dir << "\n";

    try
    {
        auto img_map = colmap::read_images_bin(colmap_dir + "/images.bin");

        if (!colmap::check_image_paths(img_map, img_folder))
            std::cerr << "[Gloc] WARNING: some map images are missing on disk.\n";

        map_.calibs = colmap::read_cameras_bin(colmap_dir + "/cameras.bin");

        const Eigen::Matrix4d world_transform =
            colmap::load_world_transform(colmap_dir);

        map_.images.clear();
        map_.images.reserve(img_map.size());
        for (auto &[id, img] : img_map)
            map_.images.push_back(std::move(img));

        std::sort(map_.images.begin(), map_.images.end(),
                  [](const colmap::Image &a, const colmap::Image &b) {
                      return a.image_id < b.image_id;
                  });

        colmap::apply_world_transform(map_.images, world_transform);

        const auto rigs = colmap::read_rigs(colmap_dir + "/rigs.bin");
        map_.cam_rig_map = colmap::build_camera_rig_map(rigs);

        std::cout << "[Gloc] Map images  : " << map_.images.size() << "\n"
                  << "[Gloc] Map cameras : " << map_.calibs.size() << "\n"
                  << "[Gloc] Rig slots     : " << map_.cam_rig_map.size() << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Gloc] ERROR loading COLMAP data: " << e.what() << "\n";
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadDatabase
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::loadDatabase()
{
    std::cout << "[Gloc] Loading DBoW3 database ...\n";

    try
    {
        dbow3::load_dbow3_database(GLOC_DBOW3_DATABASE,
                                   GLOC_DBOW3_VOCAB,
                                   map_.db,
                                   map_.feats);

        std::cout << "[Gloc] DB entries  : " << map_.db.size() << "\n"
                  << "[Gloc] Map feats  : " << map_.feats.size() << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Gloc] ERROR loading DBoW3 database: " << e.what() << "\n";
        return false;
    }

    return true;
}

} // namespace gloc