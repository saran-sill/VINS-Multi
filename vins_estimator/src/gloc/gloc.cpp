#include "gloc.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>

#include "colmap_util.h"
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

    // ── Load query camera models from GLOC_CAM_MODULES ───────────────────────
    // One model per gloc module, loaded from calib_file_[0] (cam0 = gloc cam).
    // Used in runCorrespondences to undistort query keypoints via liftProjective.
    query_cameras_.clear();
    for (const auto &mod : vins_multi::GLOC_CAM_MODULES)
    {
        camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(mod.calib_file_[0]);
        if (!cam)
        {
            ROS_ERROR("[Gloc::init] Failed to load camera model from %s",
                      mod.calib_file_[0].c_str());
            return false;
        }
        query_cameras_.push_back(cam);
        std::cout << "[Gloc::init] Loaded query camera model from "
                  << mod.calib_file_[0] << "\n";
    }

    // ── Construct shared ORB extractor ───────────────────────────────────────
    {
        const int score_type = (GLOC_ORB_SCORE_TYPE == 1)
                                   ? cv::ORB::FAST_SCORE
                                   : cv::ORB::HARRIS_SCORE;

        PointFeatureExtractorORB::Parameters p(
            GLOC_ORB_NFEATURES,
            GLOC_ORB_SCALE_FACTOR,
            GLOC_ORB_NLEVELS,
            GLOC_ORB_EDGE_THRESHOLD,
            GLOC_ORB_FIRST_LEVEL,
            GLOC_ORB_WTA_K,
            score_type,
            GLOC_ORB_PATCH_SIZE,
            GLOC_ORB_FAST_THRESHOLD);

        orb_extractor_ = std::make_unique<PointFeatureExtractorORB>(p);
    }

    // ── Construct BEBLID extractor (optional) ────────────────────────────────
    if (GLOC_USE_BEBLID)
    {
        const int n_bits = (GLOC_BEBLID_N_BITS == 256)
                               ? cv::xfeatures2d::BEBLID::SIZE_256_BITS
                               : cv::xfeatures2d::BEBLID::SIZE_512_BITS;

        beblid_extractor_ = cv::xfeatures2d::BEBLID::create(GLOC_BEBLID_SCALE_FACTOR, n_bits);

        std::cout << "[Gloc::init] BEBLID extractor enabled"
                  << " (scale_factor=" << GLOC_BEBLID_SCALE_FACTOR
                  << " n_bits=" << GLOC_BEBLID_N_BITS << ").\n";
    }

    // ── Construct feature matcher ────────────────────────────────────────────
    if (GLOC_USE_GMS)
    {
        feat_matcher_ = std::make_unique<PointFeatureMatcherGMS>(cv::NORM_HAMMING,
                                                                 static_cast<bool>(GLOC_GMS_WITH_ROTATION),
                                                                 static_cast<bool>(GLOC_GMS_WITH_SCALE),
                                                                 static_cast<double>(GLOC_GMS_THRESHOLD));
        std::cout << "[Gloc::init] Matcher: GMS"
                  << " (rotation=" << GLOC_GMS_WITH_ROTATION
                  << " scale=" << GLOC_GMS_WITH_SCALE
                  << " threshold=" << GLOC_GMS_THRESHOLD << ")\n";
    }
    else
    {
        feat_matcher_ = std::make_unique<PointFeatureMatcherBruteForce>(
            cv::NORM_HAMMING);
        std::cout << "[Gloc::init] Matcher: BruteForce Hamming.\n";
    }

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

        // ── Phase 1b: update T_map_local for VINS local pose changes ─────────
        //
        // When VINS re-optimizes keyframe local poses, T_map_local becomes
        // stale. We correct it using the delta between the local poses at
        // solve time (last_snap_poses_) and the fresh ones just received.
        //
        // For each common keyframe i (exists in both old snap and new snapshot):
        //   ΔT[i] = T_local_new[i] ⊕ T_local_old[i]^-1
        //         applied as: T_map_local_new = T_map_local_old ⊕ ΔT_mean
        //
        // If no common keyframes exist, T_map_local is left unchanged — the
        // next successful gloc solve will produce a fresh estimate.
        {
            std::lock_guard<std::mutex> snap_lk(snap_mutex_);

            if (snapped_ && !last_snap_poses_.empty())
            {
                using Mat3d = Eigen::Matrix3d;
                using Vec3d = Eigen::Vector3d;
                using Quat = Eigen::Quaterniond;

                // Build a lookup of new local poses by t_kf
                std::map<double, const Snapshot::KeyframeEntry *> new_pose_map;
                for (const auto &kf : snapshot.keyframes)
                    new_pose_map[kf.t_kf] = &kf;

                // Accumulate weighted ΔT over common keyframes
                Vec3d dt_acc = Vec3d::Zero();
                Vec3d dq_acc_v = Vec3d::Zero();
                double dq_acc_w = 0.0;
                double w_total = 0.0;

                for (const auto &[t_kf, snap] : last_snap_poses_)
                {
                    auto it = new_pose_map.find(t_kf);
                    if (it == new_pose_map.end())
                        continue; // keyframe marginalized — skip

                    const auto &new_kf = *it->second;
                    const double w = snap.weight;

                    // ΔR = R_local_new * R_local_old^T
                    const Mat3d R_old = snap.R_local.toRotationMatrix();
                    const Mat3d R_new = new_kf.R_local.toRotationMatrix();
                    const Mat3d dR = R_new * R_old.transpose();

                    // Δt = P_local_new - dR * P_local_old
                    // (difference in local frame after rotation alignment)
                    const Vec3d dt = new_kf.P_local - dR * snap.P_local;

                    dt_acc += w * dt;
                    w_total += w;

                    Quat dq(dR);
                    if (w_total > w)
                    {
                        const Quat q_so_far(dq_acc_w / (w_total - w),
                                            dq_acc_v.x() / (w_total - w),
                                            dq_acc_v.y() / (w_total - w),
                                            dq_acc_v.z() / (w_total - w));
                        if (dq.dot(q_so_far) < 0.0)
                            dq.coeffs() = -dq.coeffs();
                    }
                    dq_acc_v += w * dq.vec();
                    dq_acc_w += w * dq.w();
                }

                if (w_total > 0.0)
                {
                    Quat dq_mean(dq_acc_w / w_total,
                                 dq_acc_v.x() / w_total,
                                 dq_acc_v.y() / w_total,
                                 dq_acc_v.z() / w_total);
                    dq_mean.normalize();
                    const Mat3d dR_mean = dq_mean.toRotationMatrix();
                    const Vec3d dt_mean = dt_acc / w_total;

                    // T_map_local_new = T_map_local_old ⊕ ΔT_mean
                    // X_world = R_map * X_local + t_map
                    // After delta: X_world = R_map * (dR * X_local_old + dt) + t_map
                    //            = R_map * dR * X_local_new_approx
                    // Simplified: just right-compose the delta
                    T_map_local_t_ = T_map_local_t_ + T_map_local_R_ * dt_mean;
                    T_map_local_R_ = T_map_local_R_ * dR_mean;

                    // Update common keyframes to new local poses;
                    // erase marginalized ones (no longer in window).
                    for (auto it = last_snap_poses_.begin();
                         it != last_snap_poses_.end();)
                    {
                        auto nit = new_pose_map.find(it->first);
                        if (nit == new_pose_map.end())
                        {
                            it = last_snap_poses_.erase(it); // marginalized
                        }
                        else
                        {
                            it->second.R_local = nit->second->R_local;
                            it->second.P_local = nit->second->P_local;
                            ++it;
                        }
                    }

                    // Fire callback with updated T_map_local
                    TMapLocalCallback cb_copy;
                    {
                        std::lock_guard<std::mutex> cb_lk(cb_mutex_);
                        cb_copy = callback_;
                    }
                    if (cb_copy)
                        cb_copy(T_map_local_R_, T_map_local_t_);
                }
            }
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

        if (working_set.empty())
            continue;

        runOrbAndDbow(working_set);
        runConsensusVoting(working_set);
        runCorrespondences(working_set);
        writeBackToStateMap(working_set);

        runOptimization(working_set);
    }

    std::cout << "[Gloc] processLoop exiting.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// runOrbAndDbow
//
// Stage 1a: for every Found slot that hasn't been processed yet, convert the
// cached image to grayscale, extract ORB features, and query DBoW3 to populate
// dbow_candidates (sorted descending by score, capped at GLOC_DBOW3_MAX_RESULTS).
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::runOrbAndDbow(std::vector<KeyframeGlocState> &working_set)
{
    for (auto &kf_state : working_set)
    {
        for (std::size_t g = 0; g < kf_state.per_gloc.size(); ++g)
        {
            auto &slot = kf_state.per_gloc[g];

            if (slot.verdict != LookupVerdict::Found)
                continue;
            if (slot.pipeline_done)
                continue;
            if (slot.image.empty())
                continue;

            // ── ORB extraction ───────────────────────────────────────────────
            cv::Mat gray;
            if (slot.image.channels() == 1)
                gray = slot.image;
            else
                cv::cvtColor(slot.image, gray, cv::COLOR_BGR2GRAY);

            slot.query_feats.image_size = gray.size();
            orb_extractor_->extract(gray,
                                    slot.query_feats.keypoints,
                                    &slot.query_feats.orb_descriptors);

            if (slot.query_feats.orb_descriptors.empty())
            {
                ROS_WARN_THROTTLE(2.0,
                                  "[Gloc] ORB extraction yielded no descriptors "
                                  "for keyframe t=%.4f module=%zu",
                                  kf_state.t_kf, g);
                slot.pipeline_done = true;
                continue;
            }

            // Guarantee row-contiguous memory so DBoW3 doesn't read garbage.
            if (!slot.query_feats.orb_descriptors.isContinuous())
                slot.query_feats.orb_descriptors = slot.query_feats.orb_descriptors.clone();

            // ── BEBLID (optional) ────────────────────────────────────────────
            if (beblid_extractor_ && !slot.query_feats.keypoints.empty())
            {
                beblid_extractor_->compute(gray,
                                           slot.query_feats.keypoints,
                                           slot.query_feats.beblid_descriptors);
            }

            // ── DBoW3 query ──────────────────────────────────────────────────
            DBoW3::QueryResults dbow_ret;
            map_.db.query(slot.query_feats.orb_descriptors,
                          dbow_ret,
                          std::max(GLOC_DBOW3_MAX_RESULTS, 1),
                          /*max_id=*/-1);

            slot.dbow_candidates.clear();
            for (const DBoW3::Result &r : dbow_ret)
            {
                if (r.Score < GLOC_DBOW3_MIN_SCORE)
                    continue;
                const std::size_t train_idx = static_cast<std::size_t>(r.Id);
                if (train_idx >= map_.images.size())
                    continue;
                slot.dbow_candidates.push_back({r.Score, train_idx});
            }

            std::sort(slot.dbow_candidates.begin(),
                      slot.dbow_candidates.end(),
                      [](const auto &a, const auto &b) {
                          return a.first > b.first;
                      });

            if (static_cast<int>(slot.dbow_candidates.size()) > GLOC_DBOW3_MAX_RESULTS)
                slot.dbow_candidates.resize(GLOC_DBOW3_MAX_RESULTS);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// runConsensusVoting
//
// Stage 1b: cross-keyframe magnitude-consistency filter.
//
// For every candidate (i, n) check how many distinct other keyframes j have at
// least one candidate m satisfying:
//
//   | ||world_pos(c_in) - world_pos(c_jm)|| - ||P_local_i - P_local_j|| |
//       < GLOC_VOTE_EPS_M
//
// Candidates that agree with fewer than min_votes_threshold other keyframes are
// rejected. Among survivors the one with the highest DBoW3 score is elected as
// best_train_idx for that slot.
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::runConsensusVoting(std::vector<KeyframeGlocState> &working_set)
{
    const int X = static_cast<int>(working_set.size());
    const int min_votes_threshold = (GLOC_VOTE_MIN_VOTES < 0) ? static_cast<int>(std::ceil((X - 1) / 2.0))
                                                              : GLOC_VOTE_MIN_VOTES;

    const std::size_t n_modules = working_set.empty() ? 0 : working_set[0].per_gloc.size();

    auto train_world_pos = [&](std::size_t train_idx) -> Eigen::Vector3d {
        const colmap::Image &img = map_.images[train_idx];
        return -(img.q_c_w.inverse() * img.t_c_w);
    };

    for (std::size_t g = 0; g < n_modules; ++g)
    {
        // votes[i][n] = number of distinct other keyframes that agree with
        // candidate n of keyframe i.
        std::vector<std::vector<int>> votes(X);
        for (int i = 0; i < X; ++i)
            votes[i].assign(
                working_set[i].per_gloc[g].dbow_candidates.size(), 0);

        for (int i = 0; i < X; ++i)
        {
            const auto &slot_i = working_set[i].per_gloc[g];
            if (slot_i.dbow_candidates.empty())
                continue;

            for (int ni = 0; ni < static_cast<int>(slot_i.dbow_candidates.size()); ++ni)
            {
                const Eigen::Vector3d pos_in = train_world_pos(slot_i.dbow_candidates[ni].second);

                std::set<int> agreeing_kfs;
                for (int j = 0; j < X; ++j)
                {
                    if (j == i)
                        continue;

                    const auto &slot_j = working_set[j].per_gloc[g];
                    if (slot_j.dbow_candidates.empty())
                        continue;

                    const double local_dist = (working_set[i].P_local - working_set[j].P_local).norm();

                    for (const auto &[score_jm, train_jm] : slot_j.dbow_candidates)
                    {
                        const double world_dist = (pos_in - train_world_pos(train_jm)).norm();

                        if (std::abs(world_dist - local_dist) < GLOC_VOTE_EPS_M)
                        {
                            agreeing_kfs.insert(j);
                            break; // one agreement per other keyframe is enough
                        }
                    }
                }

                votes[i][ni] = static_cast<int>(agreeing_kfs.size());
            }
        }

        // Elect the highest-score surviving candidate per keyframe.
        for (int i = 0; i < X; ++i)
        {
            auto &slot = working_set[i].per_gloc[g];

            int best_ni = -1;
            double best_score = -1.0;

            for (int ni = 0; ni < static_cast<int>(slot.dbow_candidates.size()); ++ni)
            {
                if (votes[i][ni] < min_votes_threshold)
                    continue;

                const double score = slot.dbow_candidates[ni].first;
                if (score > best_score)
                {
                    best_score = score;
                    best_ni = ni;
                }
            }

            if (best_ni >= 0)
            {
                slot.best_train_idx = static_cast<int>(slot.dbow_candidates[best_ni].second);
                ROS_DEBUG("[Gloc] kf t=%.4f module=%zu → train_idx=%d "
                          "(votes=%d score=%.4f)",
                          working_set[i].t_kf, g, slot.best_train_idx,
                          votes[i][best_ni], best_score);
            }
            else
            {
                slot.best_train_idx = -1;
                ROS_DEBUG("[Gloc] kf t=%.4f module=%zu → rejected by voting",
                          working_set[i].t_kf, g);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// subsampleMatchesMinDist  (file-local helper)
//
// Greedy spatial subsampling: sorts matches by descriptor distance (best
// first) then keeps a match only if its query keypoint is at least
// min_dist_px pixels away from every already-kept keypoint.  This spreads
// correspondences across the image and removes redundant clustered matches
// before geometric verification.
// ─────────────────────────────────────────────────────────────────────────────

static void subsampleMatchesMinDist(const std::vector<cv::KeyPoint> &keypoints0,
                                    const std::vector<cv::DMatch> &matches_in,
                                    std::vector<cv::DMatch> &matches_out,
                                    const float min_dist_px)
{
    matches_out.clear();

    std::vector<cv::DMatch> sorted = matches_in;
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) {
                  return a.distance < b.distance;
              });

    std::vector<cv::Point2f> kept;
    kept.reserve(sorted.size());

    for (const auto &m : sorted)
    {
        const cv::Point2f &pt = keypoints0[m.queryIdx].pt;

        bool too_close = false;
        for (const auto &k : kept)
        {
            const float dx = pt.x - k.x;
            const float dy = pt.y - k.y;
            if (dx * dx + dy * dy < min_dist_px * min_dist_px)
            {
                too_close = true;
                break;
            }
        }

        if (!too_close)
        {
            matches_out.push_back(m);
            kept.push_back(pt);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// runCorrespondences
//
// Stage 1c: for each slot with a valid best_train_idx, match query ORB
// descriptors against the pre-cached train features, apply geometric
// verification (RANSAC fundamental matrix), and store the surviving 2D-2D
// point pairs (both distorted and undistorted).  Slots that fail any check
// have best_train_idx reset to -1 and are marked pipeline_done.
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::runCorrespondences(std::vector<KeyframeGlocState> &working_set)
{
    for (auto &kf_state : working_set)
    {
        for (std::size_t g = 0; g < kf_state.per_gloc.size(); ++g)
        {
            auto &slot = kf_state.per_gloc[g];

            if (slot.pipeline_done)
                continue;
            if (slot.verdict != LookupVerdict::Found)
                continue;
            if (slot.best_train_idx < 0)
            {
                slot.pipeline_done = true;
                continue;
            }

            const std::size_t ti = static_cast<std::size_t>(slot.best_train_idx);

            if (ti >= map_.feats.size() ||
                map_.feats[ti].orb_descriptors.empty())
            {
                ROS_WARN_THROTTLE(2.0,
                                  "[Gloc] train_idx=%zu has no cached features "
                                  "— skipping",
                                  ti);
                slot.pipeline_done = true;
                continue;
            }

            const dbow3::ImageFeatures &train_feats = map_.feats[ti];

            // ── Raw descriptor matching ──────────────────────────────────────
            //
            // Pick descriptor type based on config. Fall back to ORB if BEBLID
            // was requested but the train image has no beblid_descriptors cached
            // (e.g. the feature cache was built without BEBLID).
            const bool use_beblid = GLOC_USE_BEBLID &&
                                    !slot.query_feats.beblid_descriptors.empty() &&
                                    !train_feats.beblid_descriptors.empty();

            const cv::Mat &desc0 = use_beblid ? slot.query_feats.beblid_descriptors
                                              : slot.query_feats.orb_descriptors;
            const cv::Mat &desc1 = use_beblid ? train_feats.beblid_descriptors
                                              : train_feats.orb_descriptors;

            std::vector<cv::DMatch> matches;
            auto *gms = dynamic_cast<PointFeatureMatcherGMS *>(feat_matcher_.get());
            if (gms)
            {
                gms->matchGMS(slot.query_feats.image_size,
                              train_feats.image_size,
                              slot.query_feats.keypoints,
                              train_feats.keypoints,
                              desc0, desc1,
                              matches,
                              GLOC_MATCH_MAX_DIST);
            }
            else
            {
                feat_matcher_->robustMatch(desc0, desc1, matches,
                                           GLOC_MATCH_LOWE_RATIO,
                                           GLOC_MATCH_MAX_DIST);
            }

            if (matches.empty())
            {
                slot.pipeline_done = true;
                continue;
            }

            // ── Spatial subsampling (optional) ───────────────────────────────
            if (GLOC_SUBSAMPLE_MIN_DIST_PX > 0.0f)
            {
                std::vector<cv::DMatch> spread;
                subsampleMatchesMinDist(slot.query_feats.keypoints,
                                        matches, spread,
                                        GLOC_SUBSAMPLE_MIN_DIST_PX);
                matches.swap(spread);
            }

            if (matches.empty())
            {
                slot.pipeline_done = true;
                continue;
            }

            // ── Undistort matched keypoints ──────────────────────────────────
            //
            // Query: liftProjective via the camodocal model loaded from
            //        GLOC_CAM_MODULES[g].calib_file_[0] — handles any distortion
            //        model (pinhole, equidistant, scaramuzza, etc.).
            // Train: colmap::undistort_point with the COLMAP map calibration.
            auto tcal_it = map_.calibs.find(map_.images[ti].camera_id);
            if (tcal_it == map_.calibs.end())
            {
                ROS_WARN_THROTTLE(2.0,
                                  "[Gloc] no calibration for train camera_id=%u",
                                  map_.images[ti].camera_id);
                slot.pipeline_done = true;
                continue;
            }
            const colmap::CameraCalib &train_cal = tcal_it->second;

            if (g >= query_cameras_.size() || !query_cameras_[g])
            {
                ROS_WARN_THROTTLE(2.0,
                                  "[Gloc] no query camera model for module=%zu", g);
                slot.pipeline_done = true;
                continue;
            }
            const camodocal::CameraPtr &query_cam = query_cameras_[g];

            const auto &kp0 = slot.query_feats.keypoints;
            const auto &kp1 = train_feats.keypoints;
            std::vector<cv::KeyPoint> undist_kp0(kp0.size());
            std::vector<cv::KeyPoint> undist_kp1(kp1.size());

            // Virtual-camera principal point for each side.
            // Following the same convention as FeatureTracker::rejectWithF,
            // both sides are projected into a synthetic camera with
            // fx=fy=FOCAL_LENGTH and principal point at image centre.
            // This puts query and train in the same consistent pixel space
            // regardless of their actual intrinsics.
            const double query_cx = slot.query_feats.image_size.width / 2.0;
            const double query_cy = slot.query_feats.image_size.height / 2.0;
            const double train_cx = train_cal.width / 2.0;
            const double train_cy = train_cal.height / 2.0;

            for (const auto &m : matches)
            {
                // ── Query: liftProjective → virtual camera ───────────────────
                Eigen::Vector3d ray;
                query_cam->liftProjective(Eigen::Vector2d(kp0[m.queryIdx].pt.x, kp0[m.queryIdx].pt.y), ray);
                undist_kp0[m.queryIdx].pt = cv::Point2f((float)(vins_multi::FOCAL_LENGTH * ray.x() / ray.z() + query_cx),
                                                        (float)(vins_multi::FOCAL_LENGTH * ray.y() / ray.z() + query_cy));

                // ── Train: colmap undistort → normalised → virtual camera ────
                const Eigen::Vector2d u1 = colmap::undistort_point(Eigen::Vector2d(kp1[m.trainIdx].pt.x, kp1[m.trainIdx].pt.y), train_cal);
                const double xn = (u1.x() - train_cal.cx) / train_cal.fx;
                const double yn = (u1.y() - train_cal.cy) / train_cal.fy;
                undist_kp1[m.trainIdx].pt = cv::Point2f((float)(vins_multi::FOCAL_LENGTH * xn + train_cx),
                                                        (float)(vins_multi::FOCAL_LENGTH * yn + train_cy));
            }

            // ── Geometric verification ───────────────────────────────────────
            PointFeatureMatcher::geometricTest(undist_kp0, undist_kp1, matches,
                                               GLOC_MATCH_GEOM_REPROJ_TH,
                                               GLOC_MATCH_GEOM_CONFIDENCE,
                                               GLOC_MATCH_GEOM_SAMPSON_SQ);

            if (static_cast<int>(matches.size()) < GLOC_MATCH_MIN_INLIERS)
            {
                ROS_DEBUG("[Gloc] kf t=%.4f module=%zu train_idx=%zu: "
                          "only %zu inliers after geom test (need %d) — rejecting",
                          kf_state.t_kf, g, ti,
                          matches.size(), GLOC_MATCH_MIN_INLIERS);
                slot.best_train_idx = -1;
                slot.pipeline_done = true;
                continue;
            }

            // ── Store correspondences ────────────────────────────────────────
            PointFeatureMatcher::matchesToPointCorrespondences(kp0, kp1, matches, slot.pt_pairs_distorted, 1.0f);
            PointFeatureMatcher::matchesToPointCorrespondences(undist_kp0, undist_kp1, matches, slot.pt_pairs_undistorted, 1.0f);

            slot.pipeline_done = true;

            ROS_DEBUG("[Gloc] kf t=%.4f module=%zu → %zu correspondences "
                      "with train_idx=%zu",
                      kf_state.t_kf, g, slot.pt_pairs_distorted.size(), ti);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// writeBackToStateMap
//
// Stage 1d: re-acquire state_mutex_ briefly and flush pipeline_done slots from
// the working_set copy back into state_map_. Keyframes that were marginalised
// by the estimator between the working_set copy and this write-back are silently
// skipped (they're no longer in state_map_).
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::writeBackToStateMap(const std::vector<KeyframeGlocState> &working_set)
{
    std::lock_guard<std::mutex> lk(state_mutex_);

    for (const auto &kf_state : working_set)
    {
        auto it = state_map_.find(kf_state.t_kf);
        if (it == state_map_.end())
            continue; // marginalised between copy and write-back

        for (std::size_t g = 0; g < kf_state.per_gloc.size(); ++g)
        {
            const auto &src = kf_state.per_gloc[g];
            if (!src.pipeline_done)
                continue;

            auto &dst = it->second.per_gloc[g];
            dst.query_feats = src.query_feats;
            dst.dbow_candidates = src.dbow_candidates;
            dst.best_train_idx = src.best_train_idx;
            dst.pt_pairs_distorted = src.pt_pairs_distorted;
            dst.pt_pairs_undistorted = src.pt_pairs_undistorted;
            dst.pipeline_done = true;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// runOptimization
//
// Stage 2: joint Ceres optimization over all keyframes in working_set.
//
// Coordinate conventions (see also gloc_cost_functors.h):
//
//   R_local, P_local  represent  T_local_body:
//     X_local = R_local * X_body + P_local
//     R_local ≡ R_local_body  (rotation body → local)
//     P_local ≡ t_local_body  (position of body origin in local frame)
//
//   Optimisation variable per keyframe i:
//     omega_i[3]  axis-angle of R_body_world[i]  (world → body)
//     t_i[3]      t_world_body[i]  (position of body in world frame)
//
//   T_map_local convention (output):
//     X_world = R_map_local * X_local + t_map_local
//
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::runOptimization(std::vector<KeyframeGlocState> &working_set)
{
    using Mat3d = Eigen::Matrix3d;
    using Vec3d = Eigen::Vector3d;
    using Quat = Eigen::Quaterniond;

    const int X = static_cast<int>(working_set.size());

    // ── 1. Guard: count valid slots ───────────────────────────────────────────
    int valid_slot_count = 0;
    for (const auto &kf : working_set)
        for (const auto &slot : kf.per_gloc)
            if (slot.pipeline_done && slot.best_train_idx >= 0 &&
                !slot.pt_pairs_undistorted.empty())
                ++valid_slot_count;

    if (valid_slot_count < GLOC_MIN_PAIRS)
    {
        ROS_DEBUG("[Gloc::runOptimization] Only %d valid slots (need %d) — skip",
                  valid_slot_count, GLOC_MIN_PAIRS);
        return false;
    }

    // ── 2. Init variables ─────────────────────────────────────────────────────
    // omega_kf[i][3]: axis-angle R_body_world[i]  (world → body)
    // t_kf[i][3]:     t_world_body[i]
    std::vector<std::array<double, 3>> omega_kf(X);
    std::vector<std::array<double, 3>> t_kf(X);

    for (int i = 0; i < X; ++i)
    {
        const auto &kf = working_set[i];

        bool snapped_local;
        Mat3d R_map_local;
        Vec3d t_map_local;
        {
            std::lock_guard<std::mutex> lk(snap_mutex_);
            snapped_local = snapped_;
            R_map_local = T_map_local_R_;
            t_map_local = T_map_local_t_;
        }

        if (snapped_local)
        {
            // Derive T_map_body[i] from last known T_map_local:
            //   R_world_body[i] = T_map_local_R_ * R_local_body[i]
            //   t_world_body[i] = T_map_local_R_ * P_local[i] + T_map_local_t_
            const Mat3d R_local_body = kf.R_local.toRotationMatrix();
            const Mat3d R_world_body = R_map_local * R_local_body;
            const Vec3d t_world_body = R_map_local * kf.P_local + t_map_local;

            const Eigen::AngleAxisd aa(R_world_body);
            const Vec3d ov = aa.axis() * aa.angle();
            omega_kf[i] = {ov.x(), ov.y(), ov.z()};
            t_kf[i] = {t_world_body.x(), t_world_body.y(), t_world_body.z()};
        }
        else
        {
            // No prior — use train image rotation as rough seed, VINS position
            // as translation seed. Yaw candidates in init pass will correct rotation.
            const auto &slot0 = kf.per_gloc[0];
            if (slot0.best_train_idx >= 0)
            {
                const colmap::Image &train_img =
                    map_.images[static_cast<size_t>(slot0.best_train_idx)];
                const Eigen::AngleAxisd aa(train_img.q_c_w.toRotationMatrix());
                const Vec3d ov = aa.axis() * aa.angle();
                omega_kf[i] = {ov.x(), ov.y(), ov.z()};
            }
            else
            {
                omega_kf[i] = {0.0, 0.0, 0.0};
            }
            t_kf[i] = {kf.P_local.x(), kf.P_local.y(), kf.P_local.z()};
        }
    }

    // ── 3. Flatten observations ───────────────────────────────────────────────
    struct FlatObs
    {
        GlocReprojCost rep;
        GlocEpipolarCost epi;
        int kf_idx;
        int mod_idx;
    };

    std::vector<FlatObs> flat_obs;
    flat_obs.reserve(4096);
    std::vector<double> rhos;
    rhos.reserve(4096);

    const double kMaxDepthM = GLOC_MAX_DEPTH_M;

    for (int i = 0; i < X; ++i)
    {
        const auto &kf = working_set[i];
        for (std::size_t g = 0; g < kf.per_gloc.size(); ++g)
        {
            const auto &slot = kf.per_gloc[g];
            if (!slot.pipeline_done || slot.best_train_idx < 0 ||
                slot.pt_pairs_undistorted.empty())
                continue;

            const std::size_t ti = static_cast<std::size_t>(slot.best_train_idx);
            const colmap::Image &train_img = map_.images[ti];
            const colmap::CameraCalib &train_cal = map_.calibs.at(train_img.camera_id);

            const Mat3d R_j = train_img.q_c_w.toRotationMatrix();
            const Vec3d t_j = train_img.t_c_w;
            const Vec3d o_j = -(R_j.transpose() * t_j); // train centre in world

            // Cam extrinsic: R_cam_body, t_cam_body from imu_T_cam
            const Mat3d R_cb = vins_multi::GLOC_CAM_MODULES[g].ric_[0].toRotationMatrix();
            const Vec3d t_cb = vins_multi::GLOC_CAM_MODULES[g].tic_[0];

            // Virtual camera intrinsics
            const double fx_q = vins_multi::FOCAL_LENGTH;
            const double fy_q = vins_multi::FOCAL_LENGTH;
            const double cx_q = slot.query_feats.image_size.width / 2.0;
            const double cy_q = slot.query_feats.image_size.height / 2.0;

            // Flatten to row-major arrays
            double R_j_arr[9], Rcr_arr[9], t_j_arr[3], tcr_arr[3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                {
                    R_j_arr[r * 3 + c] = R_j(r, c);
                    Rcr_arr[r * 3 + c] = R_cb(r, c);
                }
            t_j_arr[0] = t_j.x();
            t_j_arr[1] = t_j.y();
            t_j_arr[2] = t_j.z();
            tcr_arr[0] = t_cb.x();
            tcr_arr[1] = t_cb.y();
            tcr_arr[2] = t_cb.z();

            for (const auto &[pq, pt] : slot.pt_pairs_undistorted)
            {
                // Train normalised bearing from COLMAP calibration
                const Vec3d m_t((pt.x() - train_cal.cx) / train_cal.fx,
                                (pt.y() - train_cal.cy) / train_cal.fy,
                                1.0);
                const Vec3d Rtm = R_j.transpose() * m_t;

                // ── Reprojection functor ─────────────────────────────────────
                GlocReprojCost rep{};
                rep.Rtm[0] = Rtm.x();
                rep.Rtm[1] = Rtm.y();
                rep.Rtm[2] = Rtm.z();
                rep.oj[0] = o_j.x();
                rep.oj[1] = o_j.y();
                rep.oj[2] = o_j.z();
                rep.pq[0] = pq.x();
                rep.pq[1] = pq.y();
                std::memcpy(rep.Rcr, Rcr_arr, sizeof(Rcr_arr));
                std::memcpy(rep.tcr, tcr_arr, sizeof(tcr_arr));
                rep.fx = fx_q;
                rep.fy = fy_q;
                rep.cx_ = cx_q;
                rep.cy_ = cy_q;

                // ── Epipolar functor ─────────────────────────────────────────
                GlocEpipolarCost epi{};
                epi.x_q[0] = (pq.x() - cx_q) / fx_q;
                epi.x_q[1] = (pq.y() - cy_q) / fy_q;
                epi.x_q[2] = 1.0;
                epi.x_t[0] = m_t.x();
                epi.x_t[1] = m_t.y();
                epi.x_t[2] = m_t.z();
                std::memcpy(epi.R_j, R_j_arr, sizeof(R_j_arr));
                epi.t_j[0] = t_j_arr[0];
                epi.t_j[1] = t_j_arr[1];
                epi.t_j[2] = t_j_arr[2];
                std::memcpy(epi.Rcr, Rcr_arr, sizeof(Rcr_arr));
                std::memcpy(epi.tcr, tcr_arr, sizeof(tcr_arr));
                epi.scale = std::sqrt(fx_q * fy_q);

                flat_obs.push_back({rep, epi, i, static_cast<int>(g)});
                rhos.push_back(0.1);
            }
        }
    }

    const int N = static_cast<int>(flat_obs.size());
    if (N < GLOC_MIN_PAIRS)
    {
        ROS_DEBUG("[Gloc::runOptimization] Too few observations (%d) — skip", N);
        return false;
    }

    ROS_DEBUG("[Gloc::runOptimization] X=%d keyframes, N=%d observations", X, N);

    // ── Solver base options ───────────────────────────────────────────────────
    ceres::Solver::Options solver_opts;
    solver_opts.minimizer_type = ceres::TRUST_REGION;
    solver_opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_opts.linear_solver_type = (N <= 2000) ? ceres::DENSE_SCHUR
                                                 : ceres::ITERATIVE_SCHUR;
    solver_opts.preconditioner_type = (N <= 2000) ? ceres::JACOBI
                                                  : ceres::SCHUR_JACOBI;
    solver_opts.function_tolerance = 1e-8;
    solver_opts.gradient_tolerance = 1e-10;
    solver_opts.parameter_tolerance = 1e-8;
    solver_opts.minimizer_progress_to_stdout = false;
    solver_opts.num_threads = 1;

    ceres::HuberLoss huber_loss(GLOC_HUBER_DELTA);
    ceres::Problem::Options prob_opts;
    prob_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    // ── Problem builder ───────────────────────────────────────────────────────
    auto build_problem = [&](ceres::Problem &prob,
                             ceres::ParameterBlockOrdering *ord,
                             bool add_epi,
                             bool add_rel,
                             bool add_prior) {
        for (int k = 0; k < N; ++k)
        {
            const int i = flat_obs[k].kf_idx;
            double *om = omega_kf[i].data();
            double *ti = t_kf[i].data();

            if (GLOC_W_REPROJ > 0.0)
            {
                auto *cost = new ceres::AutoDiffCostFunction<GlocReprojCost, 2, 3, 3, 1>(
                    new GlocReprojCost(flat_obs[k].rep));
                auto *scaled = new ceres::ScaledLoss(
                    &huber_loss, GLOC_W_REPROJ, ceres::DO_NOT_TAKE_OWNERSHIP);
                prob.AddResidualBlock(cost, scaled, om, ti, &rhos[k]);
                prob.SetParameterLowerBound(&rhos[k], 0, 1.0 / kMaxDepthM);
                ord->AddElementToGroup(&rhos[k], 0); // group 0: eliminated first (Schur)
            }

            if (add_epi && GLOC_W_EPIPOLAR > 0.0)
            {
                auto *cost = new ceres::AutoDiffCostFunction<GlocEpipolarCost, 1, 3, 3>(
                    new GlocEpipolarCost(flat_obs[k].epi));
                auto *scaled = new ceres::ScaledLoss(
                    &huber_loss, GLOC_W_EPIPOLAR, ceres::DO_NOT_TAKE_OWNERSHIP);
                prob.AddResidualBlock(cost, scaled, om, ti);
            }

            ord->AddElementToGroup(om, 1); // group 1: pose variables
            ord->AddElementToGroup(ti, 1);
        }

        // Relative local pose between adjacent keyframes
        if (add_rel && GLOC_W_REL_POSE > 0.0)
        {
            for (int i = 0; i < X; ++i)
            {
                const int k_max = std::min(i + GLOC_REL_POSE_K, X - 1);
                for (int j = i + 1; j <= k_max; ++j)
                {
                    const auto &kf_i = working_set[i];
                    const auto &kf_j = working_set[j];

                    // Relative pose from VINS local frame:
                    //   R_rel = R_local_body[i]^T * R_local_body[j]
                    //   t_rel = R_local_body[i]^T * (P_local[j] - P_local[i])
                    const Mat3d R_li = kf_i.R_local.toRotationMatrix();
                    const Mat3d R_lj = kf_j.R_local.toRotationMatrix();
                    const Mat3d R_rel = R_li.transpose() * R_lj;
                    const Vec3d t_rel = R_li.transpose() * (kf_j.P_local - kf_i.P_local);

                    GlocRelPoseCost c{};
                    for (int r = 0; r < 3; ++r)
                        for (int col = 0; col < 3; ++col)
                            c.R_rel_local[r * 3 + col] = R_rel(r, col);
                    c.t_rel_local[0] = t_rel.x();
                    c.t_rel_local[1] = t_rel.y();
                    c.t_rel_local[2] = t_rel.z();

                    auto *cost = new ceres::AutoDiffCostFunction<GlocRelPoseCost, 6, 3, 3, 3, 3>(
                        new GlocRelPoseCost(c));
                    auto *scaled = new ceres::ScaledLoss(
                        nullptr, GLOC_W_REL_POSE, ceres::DO_NOT_TAKE_OWNERSHIP);
                    prob.AddResidualBlock(cost, scaled,
                                          omega_kf[i].data(), t_kf[i].data(),
                                          omega_kf[j].data(), t_kf[j].data());
                }
            }
        }

        // World prior from last snapped T_map_local
        if (add_prior && GLOC_W_WORLD_PRIOR > 0.0)
        {
            std::lock_guard<std::mutex> lk(snap_mutex_);
            if (!snapped_)
                return;

            for (int i = 0; i < X; ++i)
            {
                const auto &kf = working_set[i];
                const Mat3d R_local_body = kf.R_local.toRotationMatrix();

                // T_map_body_prior[i]:
                //   R_prior = R_map_local * R_local_body[i]
                //   t_prior = R_map_local * P_local[i] + t_map_local
                const Mat3d R_prior = T_map_local_R_ * R_local_body;
                const Vec3d t_prior = T_map_local_R_ * kf.P_local + T_map_local_t_;

                GlocWorldPriorCost c{};
                for (int r = 0; r < 3; ++r)
                    for (int col = 0; col < 3; ++col)
                        c.R_prior[r * 3 + col] = R_prior(r, col);
                c.t_prior[0] = t_prior.x();
                c.t_prior[1] = t_prior.y();
                c.t_prior[2] = t_prior.z();

                auto *cost = new ceres::AutoDiffCostFunction<GlocWorldPriorCost, 6, 3, 3>(
                    new GlocWorldPriorCost(c));
                auto *scaled = new ceres::ScaledLoss(
                    nullptr, GLOC_W_WORLD_PRIOR, ceres::DO_NOT_TAKE_OWNERSHIP);
                prob.AddResidualBlock(cost, scaled,
                                      omega_kf[i].data(), t_kf[i].data());
            }
        }
    };

    // ── 4. Init pass: yaw candidates (only when not snapped) ─────────────────
    if (!snapped_)
    {
        constexpr int kNumYaw = 12;
        const double kYawStep = 2.0 * M_PI / kNumYaw;

        double best_cost = std::numeric_limits<double>::max();
        std::vector<std::array<double, 3>> best_omega = omega_kf;
        std::vector<std::array<double, 3>> best_t = t_kf;
        std::vector<double> best_rhos = rhos;

        for (int yk = 0; yk < kNumYaw; ++yk)
        {
            auto cand_omega = omega_kf;
            auto cand_t = t_kf;
            auto cand_rhos = rhos;

            // Apply yaw offset to all keyframes
            const Mat3d Rz = Eigen::AngleAxisd(yk * kYawStep, Vec3d::UnitZ())
                                 .toRotationMatrix();
            for (int i = 0; i < X; ++i)
            {
                const Eigen::Map<const Vec3d> ov(cand_omega[i].data());
                const double norm = ov.norm();
                const Mat3d R_orig = Eigen::AngleAxisd(
                                         norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                         .toRotationMatrix();
                const Eigen::AngleAxisd aa_new(Rz * R_orig);
                const Vec3d ov_new = aa_new.axis() * aa_new.angle();
                cand_omega[i] = {ov_new.x(), ov_new.y(), ov_new.z()};
            }

            ceres::Problem init_prob(prob_opts);
            auto *init_ord = new ceres::ParameterBlockOrdering;
            build_problem(init_prob, init_ord,
                          /*add_epi=*/true, /*add_rel=*/true, /*add_prior=*/false);

            ceres::Solver::Options init_opts = solver_opts;
            init_opts.linear_solver_ordering.reset(init_ord);
            init_opts.max_num_iterations = GLOC_INIT_ITERS;
            init_opts.function_tolerance = 1e-3;
            init_opts.parameter_tolerance = 1e-3;
            init_opts.gradient_tolerance = 1e-3;
            init_opts.linear_solver_type = ceres::DENSE_SCHUR;
            init_opts.preconditioner_type = ceres::JACOBI;

            ceres::Solver::Summary init_summary;
            ceres::Solve(init_opts, &init_prob, &init_summary);

            if (init_summary.final_cost < best_cost)
            {
                best_cost = init_summary.final_cost;
                best_omega = cand_omega;
                best_t = cand_t;
                best_rhos = cand_rhos;
            }
        }

        omega_kf = best_omega;
        t_kf = best_t;
        rhos = best_rhos;
    }

    // ── 5. Main solve ─────────────────────────────────────────────────────────
    ceres::Problem main_prob(prob_opts);
    auto *main_ord = new ceres::ParameterBlockOrdering;
    build_problem(main_prob, main_ord,
                  /*add_epi=*/true, /*add_rel=*/true, /*add_prior=*/true);

    ceres::Solver::Options main_opts = solver_opts;
    main_opts.linear_solver_ordering.reset(main_ord);
    main_opts.max_num_iterations = GLOC_MAX_ITERS;

    ceres::Solver::Summary main_summary;
    ceres::Solve(main_opts, &main_prob, &main_summary);

    ROS_DEBUG("[Gloc::runOptimization] %s", main_summary.BriefReport().c_str());

    // ── 6. Inlier check ───────────────────────────────────────────────────────
    int inliers = 0;
    for (int k = 0; k < N; ++k)
    {
        const int i = flat_obs[k].kf_idx;
        double res[2];
        flat_obs[k].rep(omega_kf[i].data(), t_kf[i].data(), &rhos[k], res);
        const double err = std::sqrt(res[0] * res[0] + res[1] * res[1]);
        if (err < GLOC_INLIER_THRESH_PX)
            ++inliers;
    }

    const double inlier_ratio = static_cast<double>(inliers) / N;
    ROS_DEBUG("[Gloc::runOptimization] inliers=%d/%d (%.1f%%)",
              inliers, N, inlier_ratio * 100.0);

    if (inlier_ratio < GLOC_MIN_INLIER_RATIO)
    {
        ROS_WARN_THROTTLE(2.0,
                          "[Gloc::runOptimization] Rejected: inlier ratio %.2f < %.2f",
                          inlier_ratio, GLOC_MIN_INLIER_RATIO);
        return false;
    }

    // ── 7. Compute T_map_local as weighted mean over keyframes ────────────────
    //
    // Each optimised T_map_body[i] gives an estimate of T_map_local:
    //   R_map_local[i] = R_world_body[i] * R_local_body[i]^T
    //                  = R_world_body[i] * R_body_local[i]
    //   t_map_local[i] = t_world_body[i] - R_map_local[i] * P_local[i]
    //
    // Weight = correspondence count for keyframe i (across all valid slots).
    Vec3d t_acc = Vec3d::Zero();
    Vec3d q_acc_v = Vec3d::Zero();
    double q_acc_w = 0.0;
    double w_total = 0.0;

    for (int i = 0; i < X; ++i)
    {
        const auto &kf = working_set[i];

        double w = 0.0;
        for (const auto &slot : kf.per_gloc)
            if (slot.pipeline_done && slot.best_train_idx >= 0)
                w += static_cast<double>(slot.pt_pairs_undistorted.size());
        if (w < 1.0)
            continue;

        const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
        const double norm = ov.norm();
        const Mat3d R_world_body = Eigen::AngleAxisd(
                                       norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                       .toRotationMatrix();

        // R_local_body = kf.R_local  (rotation body → local)
        const Mat3d R_local_body = kf.R_local.toRotationMatrix();
        const Mat3d R_map_local_i = R_world_body * R_local_body.transpose();
        const Vec3d t_map_local_i = Vec3d(t_kf[i][0], t_kf[i][1], t_kf[i][2]) - R_map_local_i * kf.P_local;

        t_acc += w * t_map_local_i;
        w_total += w;

        // Weighted quaternion accumulation (ensure consistent hemisphere)
        Quat q_i(R_map_local_i);
        if (w_total > w) // not first iteration
        {
            const Quat q_acc_so_far(q_acc_w / (w_total - w),
                                    q_acc_v.x() / (w_total - w),
                                    q_acc_v.y() / (w_total - w),
                                    q_acc_v.z() / (w_total - w));
            if (q_i.dot(q_acc_so_far) < 0.0)
                q_i.coeffs() = -q_i.coeffs();
        }
        q_acc_v += w * q_i.vec();
        q_acc_w += w * q_i.w();
    }

    if (w_total < 1.0)
    {
        ROS_WARN_THROTTLE(2.0, "[Gloc::runOptimization] Zero weight — skip");
        return false;
    }

    // ── 8. Update snapped state ───────────────────────────────────────────────
    Quat q_mean(q_acc_w / w_total,
                q_acc_v.x() / w_total,
                q_acc_v.y() / w_total,
                q_acc_v.z() / w_total);
    q_mean.normalize();

    {
        std::lock_guard<std::mutex> lk(snap_mutex_);

        T_map_local_R_ = q_mean.toRotationMatrix();
        T_map_local_t_ = t_acc / w_total;
        snapped_ = true;

        // Persist local poses used in this solve for delta tracking
        last_snap_poses_.clear();
        for (int i = 0; i < X; ++i)
        {
            const auto &kf = working_set[i];
            double w = 0.0;
            for (const auto &slot : kf.per_gloc)
                if (slot.pipeline_done && slot.best_train_idx >= 0)
                    w += static_cast<double>(slot.pt_pairs_undistorted.size());
            if (w < 1.0)
                continue;
            last_snap_poses_[kf.t_kf] = {kf.R_local, kf.P_local, w};
        }
    }

    ROS_INFO("[Gloc] Snapped! T_map_local t=[%.2f %.2f %.2f]",
             T_map_local_t_.x(), T_map_local_t_.y(), T_map_local_t_.z());

    // Notify registered consumer (e.g. estimator) on the gloc worker thread.
    // The callback must be lightweight — store and return.
    {
        std::lock_guard<std::mutex> lk(cb_mutex_);
        if (callback_)
            callback_(T_map_local_R_, T_map_local_t_);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// setTMapLocalCallback
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::setTMapLocalCallback(TMapLocalCallback cb)
{
    std::lock_guard<std::mutex> lk(cb_mutex_);
    callback_ = std::move(cb);
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