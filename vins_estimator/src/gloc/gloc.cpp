#include "gloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "../utility/visualization.h"
#include "colmap_util.h"
#include "estimator/parameters.h" // for vins_multi::GLOC_CAM_MODULES
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/EquidistantCamera.h"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Query image preprocessing
//
// Applied to the cached slot image before ORB extraction so the query
// appearance matches the preprocessed train images in the COLMAP map.
// All functions adapted from extract_images_from_metadata.cpp.
// ─────────────────────────────────────────────────────────────────────────────

static cv::Mat gloc_preprocessWhiteBalance(const cv::Mat &img)
{
    cv::Scalar mean = cv::mean(img);
    double avg = (mean[0] + mean[1] + mean[2]) / 3.0;
    std::vector<cv::Mat> ch;
    cv::split(img, ch);
    for (int i = 0; i < 3; ++i)
        if (mean[i] > 0)
            ch[i].convertTo(ch[i], -1, avg / mean[i]);
    cv::Mat r;
    cv::merge(ch, r);
    return r;
}

static cv::Mat gloc_preprocessGamma(const cv::Mat &img, double gamma)
{
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i)
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(
            std::pow(i / 255.0, 1.0 / gamma) * 255.0);
    cv::Mat r;
    cv::LUT(img, lut, r);
    return r;
}

static cv::Mat gloc_preprocessDenoise(const cv::Mat &img,
                                      int d, double sc, double ss)
{
    cv::Mat r;
    cv::bilateralFilter(img, r, d, sc, ss);
    cv::bilateralFilter(r.clone(), r, d, sc, ss);
    return r;
}

static cv::Mat gloc_preprocessDenoiseColor(const cv::Mat &img,
                                           float hl, float hc)
{
    cv::Mat r;
    cv::fastNlMeansDenoisingColored(img, r, hl, hc, 7, 21);
    return r;
}

static cv::Mat gloc_preprocessTonemap(const cv::Mat &img,
                                      double gamma, double highlight,
                                      double shadow_lift)
{
    cv::Mat f32;
    img.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    for (int y = 0; y < f32.rows; ++y)
    {
        cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
        for (int x = 0; x < f32.cols; ++x)
            for (int c = 0; c < 3; ++c)
            {
                float v = row[x][c];
                v = static_cast<float>(shadow_lift) + v * static_cast<float>(1.0 - shadow_lift);
                v = std::pow(std::max(v, 0.0f), static_cast<float>(gamma));
                float knee = static_cast<float>(1.0 - highlight);
                if (v > knee)
                {
                    float t = (v - knee) / static_cast<float>(highlight);
                    t = t - t * t * 0.5f;
                    v = knee + t * static_cast<float>(highlight);
                }
                row[x][c] = std::min(v, 1.0f);
            }
    }
    cv::Mat r;
    f32.convertTo(r, CV_8UC3, 255.0);
    return r;
}

static cv::Mat gloc_preprocessCLAHE(const cv::Mat &img,
                                    double clip_limit, int grid_size)
{
    cv::Mat r;
    auto clahe = cv::createCLAHE(clip_limit, cv::Size(grid_size, grid_size));
    if (img.channels() == 1)
    {
        clahe->apply(img, r);
    }
    else
    {
        cv::Mat lab;
        cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> ch;
        cv::split(lab, ch);
        clahe->apply(ch[0], ch[0]);
        cv::merge(ch, lab);
        cv::cvtColor(lab, r, cv::COLOR_Lab2BGR);
    }
    return r;
}

static cv::Mat gloc_preprocessClarity(const cv::Mat &img,
                                      double sigma, double amount)
{
    cv::Mat f32;
    img.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    cv::Mat blurred;
    cv::GaussianBlur(f32, blurred, cv::Size(0, 0), sigma);
    cv::Mat hp;
    cv::subtract(f32, blurred, hp);
    cv::Mat boosted;
    cv::addWeighted(f32, 1.0, hp, amount, 0.0, boosted);
    cv::threshold(boosted, boosted, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::max(boosted, 0.0, boosted);
    cv::Mat r;
    boosted.convertTo(r, CV_8UC3, 255.0);
    return r;
}

static cv::Mat gloc_preprocessSharpen(const cv::Mat &img,
                                      double sigma, double amount)
{
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(0, 0), sigma);
    cv::Mat r;
    cv::addWeighted(img, 1.0 + amount, blurred, -amount, 0, r);
    return r;
}

// Apply all enabled preprocessing steps in the correct order.
// Input may be grayscale or BGR; output matches input channels.
static cv::Mat gloc_preprocessImage(const cv::Mat &image)
{
    using namespace gloc; // bring GLOC_PREPROCESS_* into scope
    cv::Mat result = image;

    // Convert to BGR for colour-aware steps if needed
    bool was_gray = (image.channels() == 1);
    if (was_gray && (GLOC_PREPROCESS_WHITE_BALANCE ||
                     GLOC_PREPROCESS_DENOISE_COLOR ||
                     GLOC_PREPROCESS_TONEMAP ||
                     GLOC_PREPROCESS_CLAHE ||
                     GLOC_PREPROCESS_CLARITY))
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);

    if (GLOC_PREPROCESS_WHITE_BALANCE && result.channels() == 3)
        result = gloc_preprocessWhiteBalance(result);

    if (GLOC_PREPROCESS_DENOISE)
        result = gloc_preprocessDenoise(result,
                                        GLOC_PREPROCESS_DENOISE_D,
                                        GLOC_PREPROCESS_DENOISE_SIGMA_COLOR,
                                        GLOC_PREPROCESS_DENOISE_SIGMA_SPACE);

    if (GLOC_PREPROCESS_DENOISE_COLOR && result.channels() == 3)
        result = gloc_preprocessDenoiseColor(result,
                                             static_cast<float>(GLOC_PREPROCESS_DENOISE_H_LUMINANCE),
                                             static_cast<float>(GLOC_PREPROCESS_DENOISE_H_COLOR));

    if (GLOC_PREPROCESS_GAMMA)
        result = gloc_preprocessGamma(result, GLOC_PREPROCESS_GAMMA_VALUE);

    if (GLOC_PREPROCESS_TONEMAP && result.channels() == 3)
        result = gloc_preprocessTonemap(result,
                                        GLOC_PREPROCESS_TONEMAP_GAMMA,
                                        GLOC_PREPROCESS_TONEMAP_HIGHLIGHT,
                                        GLOC_PREPROCESS_TONEMAP_SHADOW_LIFT);

    if (GLOC_PREPROCESS_CLAHE)
        result = gloc_preprocessCLAHE(result,
                                      GLOC_PREPROCESS_CLAHE_CLIP_LIMIT,
                                      GLOC_PREPROCESS_CLAHE_GRID_SIZE);

    if (GLOC_PREPROCESS_CLARITY)
        result = gloc_preprocessClarity(result,
                                        GLOC_PREPROCESS_CLARITY_SIGMA,
                                        GLOC_PREPROCESS_CLARITY_AMOUNT);

    if (GLOC_PREPROCESS_SHARPEN)
        result = gloc_preprocessSharpen(result,
                                        GLOC_PREPROCESS_SHARPEN_SIGMA,
                                        GLOC_PREPROCESS_SHARPEN_AMOUNT);

    // Convert back to grayscale if input was grayscale
    if (was_gray && result.channels() == 3)
        cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);

    return result;
}

static void buildUndistMap(const camodocal::CameraPtr &cam,
                           int W, int H, double focal,
                           cv::Mat &map_x, cv::Mat &map_y)
{
    map_x.create(H, W, CV_32F);
    map_y.create(H, W, CV_32F);

    auto eq = boost::dynamic_pointer_cast<camodocal::EquidistantCamera>(cam);
    if (eq)
    {
        const auto &p = eq->getParameters();

        // Build a CameraCalib matching COLMAP OpenCVFisheye (model_id=5)
        // camodocal: k2,k3,k4,k5 → COLMAP dist: [k1,k2,k3,k4]
        colmap::CameraCalib cal;
        cal.model_id = 5;
        cal.fx = p.mu();  cal.fy = p.mv();
        cal.cx = p.u0();  cal.cy = p.v0();
        cal.width  = W;
        cal.height = H;
        cal.dist = {p.k2(), p.k3(), p.k4(), p.k5()};

        const double cx = W / 2.0, cy = H / 2.0;

        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
            {
                // Pixel → undistorted pixel via colmap::undistort_point
                const Eigen::Vector2d undist = colmap::undistort_point(
                    Eigen::Vector2d(c, r), cal);
                // Undistorted pixel → normalised → virtual camera
                const double xn = (undist.x() - cal.cx) / cal.fx;
                const double yn = (undist.y() - cal.cy) / cal.fy;
                map_x.at<float>(r, c) = static_cast<float>(focal * xn + cx);
                map_y.at<float>(r, c) = static_cast<float>(focal * yn + cy);
            }
    }
    else
    {
        const double cx = W / 2.0, cy = H / 2.0;
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
            {
                Eigen::Vector3d ray;
                cam->liftProjective(Eigen::Vector2d(c, r), ray);
                map_x.at<float>(r, c) = static_cast<float>(focal * ray.x() / ray.z() + cx);
                map_y.at<float>(r, c) = static_cast<float>(focal * ray.y() / ray.z() + cy);
            }
    }
}

namespace gloc
{

// ─────────────────────────────────────────────────────────────────────────────
// yawOnlyR
//
// Extracts the yaw component of a rotation matrix and returns a pure
// yaw rotation (pitch = roll = 0). Used when GLOC_USE_4DOF is true to
// ensure T_map_local_R_ is gravity-aligned before storing and broadcasting.
//
// Yaw is extracted as atan2(R(1,0), R(0,0)) — the rotation of the X axis
// projected onto the world XY plane.
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::Matrix3d yawOnlyR(const Eigen::Matrix3d &R)
{
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraRingBuffer
// ─────────────────────────────────────────────────────────────────────────────

void CameraRingBuffer::push(double t, const cv::Mat &image)
{
    std::lock_guard<std::mutex> lk(mtx_);

    // Reject out-of-order frames. The invariant that slots_ is sorted
    // ascending by timestamp keeps findNearest correct without sorting on
    // every read.
    if (!slots_.empty() && t <= slots_.back().t)
    {
        GLOC_WARN("[CameraRingBuffer] out-of-order frame dropped: t=%.6f vs newest=%.6f",
                  t, slots_.back().t);
        return;
    }

    slots_.push_back({t, image});

    // Evict from the front, respecting the protect floor.
    //
    // Images BELOW protect_floor_ are safe to evict — gloc has already
    // resolved or given up on every keyframe that could have needed them.
    //
    // Images AT or ABOVE protect_floor_ are still needed by unresolved
    // keyframes. They survive past the soft capacity_ limit. Only the
    // hard_capacity_ ceiling forces eviction of protected images (to
    // prevent unbounded growth if onSnapshotChanged stops running).
    while (slots_.size() > capacity_)
    {
        if (slots_.front().t >= protect_floor_)
        {
            // Front image is protected. Only evict if hard ceiling exceeded.
            if (slots_.size() > hard_capacity_)
            {
                GLOC_WARN("[CameraRingBuffer] hard ceiling reached (%zu > %zu), "
                          "evicting protected t=%.6f (floor=%.6f)",
                          slots_.size(), hard_capacity_,
                          slots_.front().t, protect_floor_);
                t_evict_watermark_ = std::max(t_evict_watermark_,
                                              slots_.front().t);
                slots_.pop_front();
            }
            else
            {
                break; // let buffer grow beyond soft cap
            }
        }
        else
        {
            // Below protect floor — safe to evict, but track watermark.
            t_evict_watermark_ = std::max(t_evict_watermark_,
                                          slots_.front().t);
            slots_.pop_front();
        }
    }
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
    // with t >= t_kf, then compare its predecessor too - the nearest in time
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

    // No frame within tolerance.
    const double newest_t = slots_.back().t;
    if (newest_t <= t_kf + tol)
    {
        // Buffer hasn't advanced past the tolerance window yet — the image
        // may still arrive.
        r.verdict = LookupVerdict::NotYet;
        return r;
    }

    // Buffer has moved past t_kf + tol. The image is gone.
    // Distinguish a genuine stream gap from an eviction casualty.
    if (t_evict_watermark_ >= t_kf - tol)
    {
        // An image within the tolerance window was evicted by capacity
        // pressure before we could look. The camera DID produce a frame
        // near t_kf, but it was lost.
        r.verdict = LookupVerdict::Evicted;
        GLOC_WARN("[findNearest] EVICTED t_kf=%.4f tol=%.4f watermark=%.4f"
                  " -- increase image_ring_buffer_capacity",
                  t_kf, tol, t_evict_watermark_);
    }
    else
    {
        // Genuine gap — the camera stream had no image near t_kf.
        r.verdict = LookupVerdict::NoneInTolerance;
        if (best_it != slots_.end())
            GLOC_DEBUG("[findNearest] NoneInTol t_kf=%.4f nearest=%.4f dt=%.4f tol=%.4f",
                       t_kf, best_it->t, std::abs(best_it->t - t_kf), tol);
    }

    return r;
}

void CameraRingBuffer::trimOlderThan(double t)
{
    std::lock_guard<std::mutex> lk(mtx_);

    // Strict less-than: the slot at exactly t (if present) is kept.
    // This is the *informed* eviction called by Phase 3 of onSnapshotChanged.
    // These images are provably unneeded (every keyframe that could match
    // them is already resolved or marginalized), so we do NOT update
    // t_evict_watermark_ — this is not data loss.
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

void CameraRingBuffer::setProtectFloor(double floor)
{
    std::lock_guard<std::mutex> lk(mtx_);
    protect_floor_ = floor;
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
        GLOC_ERROR("[init] setEstimator() must be called before init()");
        return false;
    }

    GLOC_INFO("[init] Initialising global localiser ...");

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

    GLOC_INFO("[init] Allocated %zu camera ring buffer(s).", ring_buffers_.size());

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
            GLOC_ERROR("[init] Failed to load camera model from %s",
                       mod.calib_file_[0].c_str());
            return false;
        }
        query_cameras_.push_back(cam);
        GLOC_INFO("[init] Loaded query camera model from %s", mod.calib_file_[0].c_str());

        // Precompute undistortion map — replaces per-point liftProjective calls
        // in runOrbAndDbow with O(1) table lookup. Built once at init.
        {
            cv::Mat map_x, map_y;
            buildUndistMap(cam,
                           static_cast<int>(mod.img_width_),
                           static_cast<int>(mod.img_height_),
                           vins_multi::FOCAL_LENGTH,
                           map_x, map_y);
            undist_map_x_.push_back(std::move(map_x));
            undist_map_y_.push_back(std::move(map_y));
            GLOC_INFO("[init] Built undistortion map for g=%zu (%dx%d)",
                      query_cameras_.size() - 1,
                      static_cast<int>(mod.img_width_),
                      static_cast<int>(mod.img_height_));
        }
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

        GLOC_INFO("[init] BEBLID extractor: scale_factor=%.2f n_bits=%d", GLOC_BEBLID_SCALE_FACTOR, GLOC_BEBLID_N_BITS);
    }

    // ── Construct feature matcher ────────────────────────────────────────────
    if (GLOC_USE_GMS)
    {
        feat_matcher_ = std::make_unique<PointFeatureMatcherGMS>(cv::NORM_HAMMING,
                                                                 GLOC_GMS_WITH_ROTATION,
                                                                 GLOC_GMS_WITH_SCALE,
                                                                 static_cast<double>(GLOC_GMS_THRESHOLD));
        GLOC_INFO("[init] Matcher: GMS (rotation=%d scale=%d threshold=%.1f)", GLOC_GMS_WITH_ROTATION, GLOC_GMS_WITH_SCALE, GLOC_GMS_THRESHOLD);
    }
    else
    {
        feat_matcher_ = std::make_unique<PointFeatureMatcherBruteForce>(
            cv::NORM_HAMMING);
        GLOC_INFO("[init] Matcher: BruteForce Hamming");
    }

    GLOC_INFO("[init] Ready.");
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// start_process_thread
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::start_process_thread()
{
    process_thread_ = std::thread(&Gloc::processLoop, this);
    GLOC_INFO("[start] Process thread started.");
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
        GLOC_WARN("[pushCameraImage] gloc_unique_id %u out of range (size=%zu)",
                  gloc_unique_id, ring_buffers_.size());
        return;
    }

    // Store as grayscale — ORB extraction and debug saving both work on
    // grayscale. Converting here cuts ring-buffer RAM usage by 3×
    // (1920×1200×1 byte instead of ×3 bytes per slot).
    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    ring_buffers_[gloc_unique_id]->push(t, gray);
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

        // Record the sensor time of the very first snapshot for startup delay.
        if (first_snapshot_time_ < 0.0 && !snapshot.keyframes.empty())
            first_snapshot_time_ = snapshot.keyframes.front().t_kf;

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
                s.t_image = kf.t_image; // raw image timestamp for ring-buffer lookup
                s.cam_unique_id = kf.cam_unique_id;
                s.per_gloc.assign(n_modules, PerModuleResolution{});
            }
            // Refresh local pose every snapshot - VINS keeps re-optimising
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
        // If no common keyframes exist, T_map_local is left unchanged - the
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
                        continue; // keyframe marginalized - skip

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
        // ring buffer.
        //
        // Ring buffers store images at their raw ROS timestamp (no td), so
        // we query with s.t_image (the raw image time) rather than t_kf
        // (which includes the dynamically-estimated td offset). This avoids
        // a domain mismatch that would cause missed lookups when td drifts.
        for (auto &[t_kf, s] : state_map_)
        {
            for (std::size_t g = 0; g < s.per_gloc.size(); ++g)
            {
                auto &slot = s.per_gloc[g];
                if (slot.isTerminal())
                    continue; // Found, NoneInTolerance, or Evicted

                const double tol_g = vins_multi::GLOC_CAM_MODULES[g].image_match_tol_s_;
                const LookupResult r = ring_buffers_[g]->findNearest(s.t_image, tol_g);

                slot.verdict = r.verdict;
                if (r.verdict == LookupVerdict::Found)
                {
                    slot.t_image = r.t_image;
                    slot.image = r.image;
                }
                else if (r.verdict == LookupVerdict::NoneInTolerance)
                {
                    GLOC_DEBUG("[snapshot] NoneInTol t_image=%.4f t_kf=%.4f tol=%.4f g=%zu",
                               s.t_image, t_kf, tol_g, g);
                }
                // Evicted: the ROS_WARN is already emitted by findNearest.
                // The slot becomes terminal — the image is unrecoverable.
            }
        }

        // ── Phase 3: smart ring buffer trim + protect floor ─────────────────
        //
        // For each ring buffer:
        //   1. Trim images that no keyframe can possibly need (informed eviction).
        //   2. Set the protect floor so push() knows what to keep.
        //
        // Uses t_image (raw timestamp, same domain as the ring buffer).
        for (std::size_t g = 0; g < ring_buffers_.size(); ++g)
        {
            double oldest_unresolved_t_image = std::numeric_limits<double>::max();
            for (const auto &[t, s] : state_map_)
            {
                const auto &slot = s.per_gloc[g];
                if (!slot.isTerminal())
                {
                    oldest_unresolved_t_image = std::min(oldest_unresolved_t_image, s.t_image);
                }
            }

            const double tol_g = vins_multi::GLOC_CAM_MODULES[g].image_match_tol_s_;

            if (oldest_unresolved_t_image < std::numeric_limits<double>::max())
            {
                const double floor = oldest_unresolved_t_image - tol_g;
                ring_buffers_[g]->trimOlderThan(floor);
                ring_buffers_[g]->setProtectFloor(floor);
            }
            else
            {
                // All slots resolved — no protection needed. Let push()
                // evict freely at the soft capacity limit.
                ring_buffers_[g]->setProtectFloor(std::numeric_limits<double>::max());
            }
        }

        // ── Phase 4: signal the worker ──────────────────────────────────────
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
    GLOC_INFO("[processLoop] Started.");

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

            // Count verdicts across all keyframes for diagnostics.
            int n_total = 0, n_found = 0, n_none = 0, n_evicted = 0, n_notyet = 0;
            for (const auto &[t, s] : state_map_)
            {
                ++n_total;
                bool has_found = false, has_none = false, has_evicted = false;
                for (const auto &slot : s.per_gloc)
                {
                    if (slot.verdict == LookupVerdict::Found)
                        has_found = true;
                    else if (slot.verdict == LookupVerdict::NoneInTolerance)
                        has_none = true;
                    else if (slot.verdict == LookupVerdict::Evicted)
                        has_evicted = true;
                }
                if (has_found)
                    ++n_found;
                else if (has_none)
                    ++n_none;
                else if (has_evicted)
                    ++n_evicted;
                else
                    ++n_notyet;
            }
            GLOC_DEBUG("[snapshot %lu] kf=%d Found=%d NoneInTol=%d Evicted=%d NotYet=%d",
                       (unsigned long)working_snapshot_id,
                       n_total, n_found, n_none, n_evicted, n_notyet);

            // Copy out only keyframes with at least one Found slot - keyframes
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
        // state_mutex_ released - the Ceres solve below runs without it.

        if (working_set.empty())
            continue;

        // ── Startup delay ─────────────────────────────────────────────────────
        // Don't run the pipeline until GLOC_STARTUP_DELAY_S seconds have
        // elapsed since the first snapshot. Gives VINS time to build a stable
        // sliding window before gloc attempts retrieval.
        if (GLOC_STARTUP_DELAY_S > 0.0)
        {
            const double current_t = working_set.back().t_kf;
            double first_t;
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                first_t = first_snapshot_time_;
            }
            if (first_t >= 0.0 && (current_t - first_t) < GLOC_STARTUP_DELAY_S)
            {
                GLOC_DEBUG("[processLoop] startup delay: %.1fs / %.1fs",
                           current_t - first_t, GLOC_STARTUP_DELAY_S);
                continue;
            }
        }
        auto _t0 = std::chrono::steady_clock::now();
        runOrbAndDbow(working_set);
        auto _t1 = std::chrono::steady_clock::now();
        runConsensusVoting(working_set);
        auto _t2 = std::chrono::steady_clock::now();

        GLOC_DEBUG("[pipeline_time] orb_dbow=%.0fms voting=%.0fms ws=%d",
                   std::chrono::duration<double, std::milli>(_t1 - _t0).count(),
                   std::chrono::duration<double, std::milli>(_t2 - _t1).count(),
                   static_cast<int>(working_set.size()));

        // ── DEBUG: visualize vote results ─────────────────────────────────────
        // Draw lines between each keyframe's local position (used directly as
        // world position) and its voted train image camera centre in world.
        if (vins_multi::pub_gloc_vote_lines.getNumSubscribers() > 0)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> vote_pairs;
            for (const auto &kf : working_set)
            {
                for (std::size_t g = 0; g < kf.per_gloc.size(); ++g)
                {
                    const auto &slot = kf.per_gloc[g];
                    if (slot.voted_train_idxs.empty())
                        continue;

                    // Query position: local pose used directly as world coords
                    const Eigen::Vector3d q_pos = kf.P_local;

                    // Draw one line per voted match
                    for (int ti : slot.voted_train_idxs)
                    {
                        const colmap::Image &train_img =
                            map_.images[static_cast<std::size_t>(ti)];
                        const Eigen::Matrix3d R_j = train_img.q_c_w.toRotationMatrix();
                        const Eigen::Vector3d o_j = -(R_j.transpose() * train_img.t_c_w);
                        vote_pairs.push_back({q_pos, o_j});
                    }
                }
            }
            vins_multi::pubGlocVoteLines(vote_pairs);
        }

        // runCorrespondences(working_set, working_snapshot_id);  // DEBUG: disabled
        writeBackToStateMap(working_set);

        // ── Correspondence line visualization (magenta, gloc/corr_lines) ─────
        // Draw magenta lines from query body position to matched train image
        // camera centre for slots that passed geometric verification.
        if (vins_multi::pub_gloc_corr_lines.getNumSubscribers() > 0)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corr_pairs;
            for (const auto &kf : working_set)
            {
                for (std::size_t g = 0; g < kf.per_gloc.size(); ++g)
                {
                    const auto &slot = kf.per_gloc[g];
                    if (!slot.pipeline_done || slot.best_train_idx < 0 ||
                        slot.pt_pairs_undistorted.empty())
                        continue;

                    const Eigen::Vector3d q_pos = kf.P_local;

                    for (int match_k = 0;
                         match_k < static_cast<int>(slot.voted_train_idxs.size()); ++match_k)
                    {
                        if (match_k >= static_cast<int>(slot.multi_pt_pairs_undistorted.size()) ||
                            slot.multi_pt_pairs_undistorted[match_k].empty())
                            continue;

                        const colmap::Image &train_img =
                            map_.images[static_cast<std::size_t>(slot.voted_train_idxs[match_k])];
                        const Eigen::Matrix3d R_j = train_img.q_c_w.toRotationMatrix();
                        const Eigen::Vector3d o_j = -(R_j.transpose() * train_img.t_c_w);
                        corr_pairs.push_back({q_pos, o_j});
                    }
                }
            }
            vins_multi::pubGlocCorrLines(corr_pairs);
        }

        // runOptimization(working_set);  // DEBUG: disabled
    }

    GLOC_INFO("[processLoop] Exiting.");
}

// ─────────────────────────────────────────────────────────────────────────────
// saveDebugImages  (file-local helper)
//
// Saves query image with keypoints and side-by-side match image to disk.
// Called at every correspondence attempt (pass or fail) when GLOC_DEBUG_FOLDER
// is set. The 'reason' suffix encodes what happened:
//   ""                  - match accepted
//   "VOTED_OUT"         - rejected by consensus voting
//   "FEW_INLIERS_<n>"   - rejected by geometric test
//   "NO_TRAIN_IMG"      - train image could not be loaded
// ─────────────────────────────────────────────────────────────────────────────

static void saveDebugImages(const std::string &base_dir,
                            uint64_t snapshot_id,
                            double t_kf,
                            std::size_t g,
                            std::size_t train_idx,
                            const cv::Mat &query_image,
                            const std::vector<cv::KeyPoint> &kp0,
                            const std::vector<cv::KeyPoint> &kp1,
                            const std::vector<cv::DMatch> &matches,
                            const std::string &reason,
                            const gloc::Map &map)
{
    if (base_dir.empty() || query_image.empty())
        return;

    // base_dir is guaranteed to end with '/' (normalized in readParameters).
    // Create round sub-folder inside the base directory.
    const std::string round_dir =
        base_dir + "round_" + std::to_string(snapshot_id) + "/";
    fs::create_directories(round_dir);

    // Base filename
    const std::string reason_str = reason.empty() ? "" : "_" + reason;
    const std::string base = round_dir + "kf_" + std::to_string(t_kf) + "_g" + std::to_string(g) + "_train" + std::to_string(train_idx) + reason_str;

    // ── Query image with keypoints ────────────────────────────────────────────
    cv::Mat dbg_query;
    if (query_image.channels() == 1)
        cv::cvtColor(query_image, dbg_query, cv::COLOR_GRAY2BGR);
    else
        dbg_query = query_image.clone();

    // All ORB keypoints in red
    for (const auto &kp : kp0)
        cv::circle(dbg_query, kp.pt, 3, cv::Scalar(0, 0, 255), 1);
    // Matched keypoints in green
    for (const auto &m : matches)
        cv::circle(dbg_query, kp0[m.queryIdx].pt, 4, cv::Scalar(0, 255, 0), 2);

    // Annotate with reason
    const std::string label = reason.empty() ? "ACCEPTED" : reason;
    cv::putText(dbg_query, label, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                reason.empty() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
    cv::putText(dbg_query,
                "matches=" + std::to_string(matches.size()),
                cv::Point(10, 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

    cv::imwrite(base + "_query.jpg", dbg_query);

    // ── Side-by-side with match lines ────────────────────────────────────────
    const colmap::Image &train_img = map.images[train_idx];
    const std::string train_path =
        GLOC_COLMAP_IMG_FOLDER + "/" + train_img.name;
    cv::Mat train_bgr = cv::imread(train_path);

    if (train_bgr.empty())
    {
        GLOC_WARN("[runOrbAndDbow] cannot load train image: %s",
                  train_path.c_str());
        return;
    }

    // Resize train to query height
    float train_scale = 1.0f;
    if (train_bgr.rows != dbg_query.rows)
    {
        train_scale = static_cast<float>(dbg_query.rows) / train_bgr.rows;
        cv::resize(train_bgr, train_bgr, cv::Size(), train_scale, train_scale);
    }

    cv::Mat sbs;
    cv::hconcat(dbg_query, train_bgr, sbs);

    const int offset = dbg_query.cols;
    for (const auto &m : matches)
    {
        const cv::Point2f &pq = kp0[m.queryIdx].pt;
        const cv::Point2f pt(kp1[m.trainIdx].pt.x * train_scale,
                             kp1[m.trainIdx].pt.y * train_scale);
        cv::line(sbs,
                 cv::Point(static_cast<int>(pq.x), static_cast<int>(pq.y)),
                 cv::Point(static_cast<int>(pt.x) + offset, static_cast<int>(pt.y)),
                 cv::Scalar(255, 0, 255), 1);
    }

    cv::imwrite(base + "_match.jpg", sbs);
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

            // ── Preprocessing ─────────────────────────────────────────────────
            slot.preprocessed_image = gloc_preprocessImage(slot.image);

            // ── ORB extraction ───────────────────────────────────────────────
            auto _orb_t0 = std::chrono::steady_clock::now();

            // ── ORB extraction ───────────────────────────────────────────────
            cv::Mat gray;
            if (slot.preprocessed_image.channels() == 1)
                gray = slot.preprocessed_image;
            else
                cv::cvtColor(slot.preprocessed_image, gray, cv::COLOR_BGR2GRAY);

            slot.query_feats.image_size = gray.size();
            orb_extractor_->extract(gray,
                                    slot.query_feats.keypoints,
                                    &slot.query_feats.orb_descriptors);

            if (slot.query_feats.orb_descriptors.empty())
            {
                GLOC_WARN("[runOrbAndDbow] no ORB descriptors: kf_t=%.4f g=%zu",
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

            // ── Cache undistorted query keypoints ────────────────────────────
            // Use precomputed undistortion map (built at init) — O(1) nearest-
            // neighbour lookup per point instead of iterative liftProjective.
            if (g < undist_map_x_.size())
            {
                const auto &kp0 = slot.query_feats.keypoints;
                const cv::Mat &mx = undist_map_x_[g];
                const cv::Mat &my = undist_map_y_[g];
                slot.undistorted_query_kps.resize(kp0.size());
                for (std::size_t qi = 0; qi < kp0.size(); ++qi)
                {
                    const int ix = std::max(0, std::min(mx.cols - 1,
                                                        static_cast<int>(kp0[qi].pt.x + 0.5f)));
                    const int iy = std::max(0, std::min(mx.rows - 1,
                                                        static_cast<int>(kp0[qi].pt.y + 0.5f)));
                    slot.undistorted_query_kps[qi].pt = cv::Point2f(
                        mx.at<float>(iy, ix),
                        my.at<float>(iy, ix));
                }
            }

            // ── DBoW3 query ──────────────────────────────────────────────────
            auto _dbow_t0 = std::chrono::steady_clock::now();
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

            auto _dbow_t1 = std::chrono::steady_clock::now();
            GLOC_DEBUG("[orb_time] kf_t=%.4f g=%zu dbow=%.0fms candidates=%zu",
                       kf_state.t_kf, g,
                       std::chrono::duration<double, std::milli>(_dbow_t1 - _dbow_t0).count(),
                       slot.dbow_candidates.size());

            // Release raw and preprocessed images immediately — they are only
            // needed for ORB extraction (done above) and debug saving (done in
            // runCorrespondences). Holding them wastes ~7MB per slot per round.
            // Debug saving in runCorrespondences uses slot.preprocessed_image;
            // if GLOC_DEBUG_FOLDER is set we keep it, otherwise release both.
            slot.image.release();
            if (GLOC_DEBUG_FOLDER.empty())
                slot.preprocessed_image.release();
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
    // ── Overview ──────────────────────────────────────────────────────────────
    //
    // Only runs on keyframes that are NOT yet pipeline_done (new keyframes).
    // pipeline_done keyframes already have valid voted_train_idxs — we never
    // re-vote them. They do contribute their dbow_candidates as voters to
    // help constrain the translation hypothesis for new keyframes.
    //
    // For each new keyframe, find the best-scoring candidate whose implied
    // translation t = train_pos - P_local is consistent with the translation
    // implied by the majority of other keyframes (done or new) via RANSAC.
    // ─────────────────────────────────────────────────────────────────────────

    const int X = static_cast<int>(working_set.size());
    const int min_inliers = (GLOC_VOTE_MIN_VOTES < 0)
                                ? static_cast<int>(std::ceil((X - 1) / 2.0))
                                : GLOC_VOTE_MIN_VOTES;

    const std::size_t n_modules = working_set.empty() ? 0 : working_set[0].per_gloc.size();

    auto train_world_pos = [&](std::size_t train_idx) -> Eigen::Vector3d {
        const colmap::Image &img = map_.images[train_idx];
        return -(img.q_c_w.inverse() * img.t_c_w);
    };

    for (std::size_t g = 0; g < n_modules; ++g)
    {
        // Collect new keyframes that need voting
        std::vector<int> new_kf_idxs;
        for (int i = 0; i < X; ++i)
        {
            const auto &slot = working_set[i].per_gloc[g];
            if (!slot.pipeline_done &&
                slot.verdict == LookupVerdict::Found &&
                !slot.dbow_candidates.empty())
                new_kf_idxs.push_back(i);
        }

        if (new_kf_idxs.empty())
            continue;

        // ── 1. Build voter pool from ALL keyframes (new + done) ───────────────
        // Done keyframes provide stable translation anchors.
        // New keyframes provide candidates to be assigned.
        struct Cand
        {
            int kf_idx;
            int cand_idx;
            Eigen::Vector3d P_local;
            Eigen::Vector3d P_world;
            double score;
        };
        std::vector<Cand> all_cands;
        for (int i = 0; i < X; ++i)
        {
            const auto &slot = working_set[i].per_gloc[g];
            if (slot.dbow_candidates.empty())
                continue;
            for (int ni = 0; ni < static_cast<int>(slot.dbow_candidates.size()); ++ni)
                all_cands.push_back({i, ni,
                                     working_set[i].P_local,
                                     train_world_pos(slot.dbow_candidates[ni].second),
                                     slot.dbow_candidates[ni].first});
        }

        // ── 1b. Pre-filter by max distance ────────────────────────────────────
        if (GLOC_VOTE_MAX_DIST_M > 0.0)
        {
            all_cands.erase(
                std::remove_if(all_cands.begin(), all_cands.end(),
                               [](const Cand &c) {
                                   return (c.P_world - c.P_local).norm() > GLOC_VOTE_MAX_DIST_M;
                               }),
                all_cands.end());
        }

        if (all_cands.empty())
        {
            GLOC_DEBUG("[vote] g=%zu all candidates rejected by max_dist filter", g);
            for (int i : new_kf_idxs)
            {
                working_set[i].per_gloc[g].best_train_idx = -1;
                working_set[i].per_gloc[g].voted_train_idxs.clear();
            }
            continue;
        }

        // ── 2. RANSAC — seed from ALL candidates, assign only new keyframes ───
        int best_inlier_count = 0;
        std::vector<int> best_assignment(X, -1); // kf_idx -> cand_idx (new kfs only)

        for (const auto &seed : all_cands)
        {
            const Eigen::Vector3d t_hyp = seed.P_world - seed.P_local;

            // Count inliers across ALL keyframes (for hypothesis strength)
            std::vector<int> assignment(X, -1);
            int inlier_count = 0;

            for (int i = 0; i < X; ++i)
            {
                const auto &slot = working_set[i].per_gloc[g];
                if (slot.dbow_candidates.empty())
                    continue;

                const Eigen::Vector3d predicted = working_set[i].P_local + t_hyp;
                int best_ni = -1;
                double best_score = -1.0;
                for (int ni = 0; ni < static_cast<int>(slot.dbow_candidates.size()); ++ni)
                {
                    const Eigen::Vector3d wp = train_world_pos(slot.dbow_candidates[ni].second);
                    if ((wp - predicted).norm() < GLOC_VOTE_EPS_M)
                    {
                        const double sc = slot.dbow_candidates[ni].first;
                        if (sc > best_score)
                        {
                            best_score = sc;
                            best_ni = ni;
                        }
                    }
                }
                if (best_ni >= 0)
                {
                    assignment[i] = best_ni;
                    ++inlier_count;
                }
            }

            if (inlier_count > best_inlier_count)
            {
                best_inlier_count = inlier_count;
                best_assignment = assignment;
            }
        }

        // ── 3. Apply assignment to NEW keyframes only ─────────────────────────
        if (best_inlier_count >= min_inliers)
        {
            GLOC_DEBUG("[vote] g=%zu hypothesis: %d/%d inliers, assigning %zu new kfs",
                       g, best_inlier_count, X, new_kf_idxs.size());

            // Winning translation from highest-scoring inlier
            Eigen::Vector3d t_win = Eigen::Vector3d::Zero();
            for (int i = 0; i < X; ++i)
                if (best_assignment[i] >= 0)
                {
                    t_win = train_world_pos(working_set[i].per_gloc[g].dbow_candidates[best_assignment[i]].second) - working_set[i].P_local;
                    break;
                }

            for (int i : new_kf_idxs)
            {
                auto &slot = working_set[i].per_gloc[g];
                slot.voted_train_idxs.clear();

                if (best_assignment[i] >= 0)
                {
                    slot.best_train_idx = static_cast<int>(
                        slot.dbow_candidates[best_assignment[i]].second);

                    const Eigen::Vector3d predicted = working_set[i].P_local + t_win;
                    std::vector<std::pair<double, int>> inlier_cands;
                    for (const auto &[sc, ti] : slot.dbow_candidates)
                        if ((train_world_pos(ti) - predicted).norm() < GLOC_VOTE_EPS_M)
                            inlier_cands.push_back({sc, static_cast<int>(ti)});

                    std::sort(inlier_cands.begin(), inlier_cands.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
                    const int n_keep = std::min(static_cast<int>(inlier_cands.size()),
                                                GLOC_VOTE_MAX_MATCHES);
                    for (int k = 0; k < n_keep; ++k)
                        slot.voted_train_idxs.push_back(inlier_cands[k].second);

                    GLOC_DEBUG("[vote] kf_t=%.4f g=%zu -> %d match(es), best train=%d score=%.4f",
                               working_set[i].t_kf, g,
                               static_cast<int>(slot.voted_train_idxs.size()),
                               slot.best_train_idx,
                               slot.dbow_candidates[best_assignment[i]].first);
                }
                else
                {
                    slot.best_train_idx = -1;
                    GLOC_DEBUG("[vote] kf_t=%.4f g=%zu -> no inlier candidate",
                               working_set[i].t_kf, g);
                }
            }
        }
        else
        {
            GLOC_DEBUG("[vote] g=%zu rejected: best inliers %d < %d",
                       g, best_inlier_count, min_inliers);
            for (int i : new_kf_idxs)
            {
                working_set[i].per_gloc[g].best_train_idx = -1;
                working_set[i].per_gloc[g].voted_train_idxs.clear();
            }
        }
    }
}
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

void Gloc::runCorrespondences(std::vector<KeyframeGlocState> &working_set,
                              uint64_t snapshot_id)
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
                // Voted out - save debug image showing the top DBoW candidate
                // (even though it was rejected) so we can inspect the match.
                if (!GLOC_DEBUG_FOLDER.empty() && !slot.image.empty() &&
                    !slot.dbow_candidates.empty())
                {
                    const std::size_t top_ti = slot.dbow_candidates[0].second;
                    const std::vector<cv::DMatch> no_matches;
                    saveDebugImages(GLOC_DEBUG_FOLDER, snapshot_id,
                                    kf_state.t_kf, g, top_ti,
                                    slot.preprocessed_image,
                                    slot.query_feats.keypoints,
                                    map_.feats[top_ti].keypoints,
                                    no_matches,
                                    "VOTED_OUT", map_);
                }
                slot.pipeline_done = true;
                continue;
            }

            const auto &kp0 = slot.query_feats.keypoints;

            // Use cached undistorted query keypoints computed in runOrbAndDbow.
            // Fall back to map lookup if cache is missing (shouldn't happen).
            std::vector<cv::KeyPoint> undist_kp0_fallback;
            const std::vector<cv::KeyPoint> *undist_kp0_ptr = nullptr;
            if (!slot.undistorted_query_kps.empty())
            {
                undist_kp0_ptr = &slot.undistorted_query_kps;
            }
            else
            {
                GLOC_WARN("[corr] undistorted_query_kps cache missing for kf_t=%.4f g=%zu - recomputing",
                          kf_state.t_kf, g);
                undist_kp0_fallback.resize(kp0.size());
                if (g < undist_map_x_.size())
                {
                    const cv::Mat &mx = undist_map_x_[g];
                    const cv::Mat &my = undist_map_y_[g];
                    for (std::size_t qi = 0; qi < kp0.size(); ++qi)
                    {
                        const int ix = std::max(0, std::min(mx.cols - 1, static_cast<int>(kp0[qi].pt.x + 0.5f)));
                        const int iy = std::max(0, std::min(mx.rows - 1, static_cast<int>(kp0[qi].pt.y + 0.5f)));
                        undist_kp0_fallback[qi].pt = cv::Point2f(mx.at<float>(iy, ix), my.at<float>(iy, ix));
                    }
                }
                undist_kp0_ptr = &undist_kp0_fallback;
            }
            const std::vector<cv::KeyPoint> &undist_kp0 = *undist_kp0_ptr;

            const bool use_beblid = GLOC_USE_BEBLID &&
                                    !slot.query_feats.beblid_descriptors.empty();

            // ── Multi-match loop over voted_train_idxs ────────────────────────
            slot.multi_pt_pairs_distorted.clear();
            slot.multi_pt_pairs_undistorted.clear();
            bool any_success = false;

            for (int match_k = 0;
                 match_k < static_cast<int>(slot.voted_train_idxs.size()); ++match_k)
            {
                const std::size_t ti =
                    static_cast<std::size_t>(slot.voted_train_idxs[match_k]);

                if (ti >= map_.feats.size() || map_.feats[ti].orb_descriptors.empty())
                {
                    GLOC_WARN("[corr] train_idx=%zu has no cached features - skip", ti);
                    slot.multi_pt_pairs_distorted.emplace_back();
                    slot.multi_pt_pairs_undistorted.emplace_back();
                    continue;
                }

                const dbow3::ImageFeatures &train_feats = map_.feats[ti];
                const bool use_beblid_this_match =
                    use_beblid && !train_feats.beblid_descriptors.empty();
                const cv::Mat &desc0_match = use_beblid_this_match
                                                 ? slot.query_feats.beblid_descriptors
                                                 : slot.query_feats.orb_descriptors;
                const cv::Mat &desc1 = use_beblid_this_match
                                           ? train_feats.beblid_descriptors
                                           : train_feats.orb_descriptors;

                // ── Time: GMS matching ────────────────────────────────────────
                auto t_gms0 = std::chrono::steady_clock::now();

                // Single-direction knnMatch + Lowe ratio — half the cost of
                // robustMatch (which does two passes for symmetric filtering).
                // Geometric verification below handles residual outliers.
                std::vector<cv::DMatch> matches;
                std::vector<std::vector<cv::DMatch>> knn_matches;
                feat_matcher_->knnMatch(desc0_match, desc1, knn_matches, 2);
                for (const auto &m : knn_matches)
                {
                    if (m.size() < 2)
                        continue;
                    if (m[0].distance < GLOC_MATCH_LOWE_RATIO * m[1].distance &&
                        m[0].distance <= GLOC_MATCH_MAX_DIST)
                        matches.push_back(m[0]);
                }

                auto t_gms1 = std::chrono::steady_clock::now();
                const int n_after_gms = static_cast<int>(matches.size());
                GLOC_DEBUG("[corr_time] kf_t=%.4f g=%zu train=%zu [%d/%d]: "
                           "gms=%.1fms q_kp=%zu t_kp=%zu after_gms=%d",
                           kf_state.t_kf, g, ti, match_k + 1,
                           static_cast<int>(slot.voted_train_idxs.size()),
                           std::chrono::duration<double, std::milli>(t_gms1 - t_gms0).count(),
                           kp0.size(), train_feats.keypoints.size(), n_after_gms);

                if (matches.empty())
                {
                    GLOC_DEBUG("[corr] kf_t=%.4f g=%zu train=%zu: 0 matches after GMS - skip",
                               kf_state.t_kf, g, ti);
                    slot.multi_pt_pairs_distorted.emplace_back();
                    slot.multi_pt_pairs_undistorted.emplace_back();
                    continue;
                }

                // ── Time: spatial subsampling ─────────────────────────────────
                auto t_sub0 = std::chrono::steady_clock::now();

                if (GLOC_SUBSAMPLE_MIN_DIST_PX > 0.0f)
                {
                    std::vector<cv::DMatch> spread;
                    subsampleMatchesMinDist(kp0, matches, spread,
                                            GLOC_SUBSAMPLE_MIN_DIST_PX);
                    matches.swap(spread);
                }

                auto t_sub1 = std::chrono::steady_clock::now();
                GLOC_DEBUG("[corr_time] kf_t=%.4f g=%zu train=%zu: "
                           "subsample=%.1fms after=%d",
                           kf_state.t_kf, g, ti,
                           std::chrono::duration<double, std::milli>(t_sub1 - t_sub0).count(),
                           static_cast<int>(matches.size()));

                if (matches.empty())
                {
                    slot.multi_pt_pairs_distorted.emplace_back();
                    slot.multi_pt_pairs_undistorted.emplace_back();
                    continue;
                }

                // ── Time: undistort train keypoints ───────────────────────────
                auto t_tu0 = std::chrono::steady_clock::now();

                auto tcal_it = map_.calibs.find(map_.images[ti].camera_id);
                if (tcal_it == map_.calibs.end())
                {
                    GLOC_WARN("[corr] no calibration for train camera_id=%u",
                              map_.images[ti].camera_id);
                    slot.multi_pt_pairs_distorted.emplace_back();
                    slot.multi_pt_pairs_undistorted.emplace_back();
                    continue;
                }
                const colmap::CameraCalib &train_cal = tcal_it->second;
                const double train_cx = train_cal.width / 2.0;
                const double train_cy = train_cal.height / 2.0;
                const auto &kp1 = train_feats.keypoints;
                std::vector<cv::KeyPoint> undist_kp1(kp1.size());
                for (const auto &m : matches)
                {
                    const Eigen::Vector2d u1 = colmap::undistort_point(
                        Eigen::Vector2d(kp1[m.trainIdx].pt.x, kp1[m.trainIdx].pt.y),
                        train_cal);
                    const double xn = (u1.x() - train_cal.cx) / train_cal.fx;
                    const double yn = (u1.y() - train_cal.cy) / train_cal.fy;
                    undist_kp1[m.trainIdx].pt = cv::Point2f(
                        (float)(vins_multi::FOCAL_LENGTH * xn + train_cx),
                        (float)(vins_multi::FOCAL_LENGTH * yn + train_cy));
                }

                auto t_tu1 = std::chrono::steady_clock::now();
                GLOC_DEBUG("[corr_time] kf_t=%.4f g=%zu train=%zu: "
                           "undistort_train=%.1fms",
                           kf_state.t_kf, g, ti,
                           std::chrono::duration<double, std::milli>(t_tu1 - t_tu0).count());

                // ── Time: geometric verification ──────────────────────────────
                auto t_geom0 = std::chrono::steady_clock::now();

                PointFeatureMatcher::geometricTest(undist_kp0, undist_kp1, matches,
                                                   GLOC_MATCH_GEOM_REPROJ_TH,
                                                   GLOC_MATCH_GEOM_CONFIDENCE,
                                                   GLOC_MATCH_GEOM_SAMPSON_SQ);

                auto t_geom1 = std::chrono::steady_clock::now();
                GLOC_DEBUG("[corr_time] kf_t=%.4f g=%zu train=%zu: "
                           "geom=%.1fms after=%d",
                           kf_state.t_kf, g, ti,
                           std::chrono::duration<double, std::milli>(t_geom1 - t_geom0).count(),
                           static_cast<int>(matches.size()));

                if (static_cast<int>(matches.size()) < GLOC_MATCH_MIN_INLIERS)
                {
                    GLOC_DEBUG("[corr] kf_t=%.4f g=%zu train=%zu: "
                               "gms=%d -> geom=%d (need %d) - REJECT",
                               kf_state.t_kf, g, ti, n_after_gms,
                               static_cast<int>(matches.size()), GLOC_MATCH_MIN_INLIERS);
                    slot.multi_pt_pairs_distorted.emplace_back();
                    slot.multi_pt_pairs_undistorted.emplace_back();
                    if (match_k == 0)
                    {
                        slot.best_train_idx = -1;
                        saveDebugImages(GLOC_DEBUG_FOLDER, snapshot_id,
                                        kf_state.t_kf, g, ti,
                                        slot.preprocessed_image, kp0, kp1, matches,
                                        "FEW_INLIERS_" + std::to_string(matches.size()),
                                        map_);
                    }
                    continue;
                }

                // Store correspondences
                std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> pd, pu;
                PointFeatureMatcher::matchesToPointCorrespondences(
                    kp0, kp1, matches, pd, 1.0f);
                PointFeatureMatcher::matchesToPointCorrespondences(
                    undist_kp0, undist_kp1, matches, pu, 1.0f);

                slot.multi_pt_pairs_distorted.push_back(pd);
                slot.multi_pt_pairs_undistorted.push_back(pu);

                GLOC_DEBUG("[corr] kf_t=%.4f g=%zu train=%zu [%d/%d] -> %zu corr",
                           kf_state.t_kf, g, ti, match_k + 1,
                           static_cast<int>(slot.voted_train_idxs.size()), pd.size());

                if (match_k == 0)
                {
                    slot.pt_pairs_distorted = pd;
                    slot.pt_pairs_undistorted = pu;
                    saveDebugImages(GLOC_DEBUG_FOLDER, snapshot_id,
                                    kf_state.t_kf, g, ti,
                                    slot.preprocessed_image, kp0, kp1, matches,
                                    "", map_);
                }
                any_success = true;
            }

            if (!any_success && slot.best_train_idx >= 0)
                slot.best_train_idx = -1;

            slot.pipeline_done = true;
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
            dst.undistorted_query_kps = src.undistorted_query_kps;
            dst.preprocessed_image = src.preprocessed_image;
            dst.dbow_candidates = src.dbow_candidates;
            dst.best_train_idx = src.best_train_idx;
            dst.voted_train_idxs = src.voted_train_idxs;
            dst.pt_pairs_distorted = src.pt_pairs_distorted;
            dst.pt_pairs_undistorted = src.pt_pairs_undistorted;
            dst.multi_pt_pairs_distorted = src.multi_pt_pairs_distorted;
            dst.multi_pt_pairs_undistorted = src.multi_pt_pairs_undistorted;

            // Free the raw and preprocessed images — they are no longer needed
            // once pipeline_done. Descriptors and dbow_candidates are kept:
            // descriptors are needed by runCorrespondences; dbow_candidates
            // are used as voters in runConsensusVoting for future rounds.
            dst.image.release();
            dst.preprocessed_image.release();
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
//     R_local ≡ R_local_body  (rotation body -> local)
//     P_local ≡ t_local_body  (position of body origin in local frame)
//
//   Optimisation variable per keyframe i:
//     omega_i[3]  axis-angle of R_body_world[i]  (world -> body)
//     t_i[3]      t_world_body[i]  (position of body in world frame)
//
//   T_map_local convention (output):
//     X_world = R_map_local * X_local + t_map_local
//
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::runOptimization_6DOF(std::vector<KeyframeGlocState> &working_set)
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

    const int min_pairs = snapped_ ? GLOC_MIN_PAIRS : GLOC_MIN_PAIRS_FIRST_SNAP;

    if (valid_slot_count < min_pairs)
    {
        GLOC_DEBUG("[opt] only %d valid slots (need %d%s) - skip",
                   valid_slot_count, min_pairs, snapped_ ? "" : " first-snap");
        return false;
    }

    // ── 2. Init variables ─────────────────────────────────────────────────────
    // omega_kf[i][3]: axis-angle of R_body_world[i]  (world -> body)
    // t_kf[i][3]:     t_world_body[i]  (body origin in world)
    //
    // Convention check (the functor uses AngleAxisRotatePoint(omega, neg_t, t_bw)):
    //   omega represents R_body_world  (world → body)
    //   t_i   represents t_world_body  (body position expressed in world)
    //
    // Snapped:   derive from T_map_local + VINS local pose.
    // Unsnapped: seed from the matched train image pose in COLMAP world.
    //            DO NOT use P_local here — that is in the VINS local frame
    //            (arbitrary origin), not in COLMAP world. Starting the
    //            optimizer there puts every residual at ~1000 px (the
    //            behind-camera clamp), the gradient is near-zero everywhere,
    //            and Ceres converges to the same wrong point after ~30 iters.

    std::vector<std::array<double, 3>> omega_kf(X);
    std::vector<std::array<double, 3>> t_kf(X);

    // Read snap state once outside the loop — snapped_ only flips on success,
    // which can't happen until after this function returns.
    bool snapped_local;
    Mat3d R_map_local;
    Vec3d t_map_local;
    {
        std::lock_guard<std::mutex> lk(snap_mutex_);
        snapped_local = snapped_;
        R_map_local = T_map_local_R_;
        t_map_local = T_map_local_t_;
    }

    // Precompute camera-body extrinsic (same for all keyframes, g=0).
    // imu_T_cam convention:  p_body = R_imu_cam * p_cam + t_imu_cam
    //   ric_[0] = R_imu_cam  (cam vectors → body/imu vectors)
    //   tic_[0] = position of cam origin in body/imu frame
    const Mat3d R_imu_cam = vins_multi::GLOC_CAM_MODULES[0].ric_[0].toRotationMatrix();
    const Vec3d t_imu_cam = vins_multi::GLOC_CAM_MODULES[0].tic_[0];

    for (int i = 0; i < X; ++i)
    {
        const auto &kf = working_set[i];

        if (snapped_local)
        {
            // ── Snapped: use T_map_local to place body in COLMAP world ────────
            //
            // R_world_body = T_map_local_R * R_local_body
            // t_world_body = T_map_local_R * P_local + T_map_local_t
            //
            // omega must encode R_body_world = R_world_body^T  (not R_world_body).
            const Mat3d R_local_body = kf.R_local.toRotationMatrix();
            const Mat3d R_world_body = R_map_local * R_local_body;
            const Vec3d t_world_body = R_map_local * kf.P_local + t_map_local;

            const Eigen::AngleAxisd aa(R_world_body.transpose()); // R_body_world
            const Vec3d ov = aa.axis() * aa.angle();
            omega_kf[i] = {ov.x(), ov.y(), ov.z()};
            t_kf[i] = {t_world_body.x(), t_world_body.y(), t_world_body.z()};
        }
        else
        {
            // ── Unsnapped: derive from matched train image pose ───────────────
            //
            // For keyframe i we use the nearest voted train image (per_gloc[0]).
            // Composition (world → cam → body):
            //
            //   R_body_world = R_imu_cam * R_cam_world
            //
            // Body position in world:
            //   oj            = train cam centre = -R_cw^T * t_cw
            //   t_world_body  = oj - R_cw^T * R_imu_cam^T * t_imu_cam
            //
            // For keyframes without a matched train image, we use {0,0,0} as
            // placeholder; a second pass below fills them with the mean of
            // the valid ones so they start in the right region of the map.
            const auto &slot0 = kf.per_gloc[0];
            if (slot0.best_train_idx >= 0)
            {
                const colmap::Image &train_img =
                    map_.images[static_cast<size_t>(slot0.best_train_idx)];

                const Mat3d R_cw = train_img.q_c_w.toRotationMatrix(); // world → cam
                const Vec3d t_cw = train_img.t_c_w;
                const Vec3d oj = -(R_cw.transpose() * t_cw); // cam centre in world

                // R_body_world = R_imu_cam * R_cam_world
                const Mat3d R_bw = R_imu_cam * R_cw;
                const Eigen::AngleAxisd aa(R_bw);
                const Vec3d ov = aa.axis() * aa.angle();
                omega_kf[i] = {ov.x(), ov.y(), ov.z()};

                // t_world_body = oj - R_world_body * t_cam_in_body
                //              = oj - R_cw^T * R_imu_cam^T * t_imu_cam
                const Vec3d t_body = oj - R_cw.transpose() * R_imu_cam.transpose() * t_imu_cam;
                t_kf[i] = {t_body.x(), t_body.y(), t_body.z()};
            }
            else
            {
                // No match — placeholder; filled below.
                omega_kf[i] = {0.0, 0.0, 0.0};
                t_kf[i] = {0.0, 0.0, 0.0}; // sentinel: will be replaced
            }
        }
    }

    // ── Backfill unmatched unsnapped keyframes ────────────────────────────────
    //
    // Keyframes without a voted train image start at (0,0,0) in world, which
    // would perturb the relative-pose constraints. Replace them with the mean
    // world position of the matched keyframes so all variables start in the
    // correct region of the COLMAP map.
    if (!snapped_local)
    {
        Vec3d t_sum = Vec3d::Zero();
        int n_valid = 0;
        for (int i = 0; i < X; ++i)
        {
            if (working_set[i].per_gloc[0].best_train_idx >= 0)
            {
                t_sum += Vec3d(t_kf[i][0], t_kf[i][1], t_kf[i][2]);
                ++n_valid;
            }
        }
        if (n_valid > 0)
        {
            const Vec3d t_mean = t_sum / n_valid;
            for (int i = 0; i < X; ++i)
            {
                if (working_set[i].per_gloc[0].best_train_idx < 0)
                    t_kf[i] = {t_mean.x(), t_mean.y(), t_mean.z()};
            }
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
                slot.multi_pt_pairs_undistorted.empty())
                continue;

            for (int match_k = 0;
                 match_k < static_cast<int>(slot.voted_train_idxs.size()); ++match_k)
            {
                if (match_k >= static_cast<int>(slot.multi_pt_pairs_undistorted.size()) ||
                    slot.multi_pt_pairs_undistorted[match_k].empty())
                    continue;

                const std::size_t ti =
                    static_cast<std::size_t>(slot.voted_train_idxs[match_k]);

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

                for (const auto &[pq, pt] : slot.multi_pt_pairs_undistorted[match_k])
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
            } // end match_k loop
        }
    }

    const int N = static_cast<int>(flat_obs.size());
    if (N < GLOC_MIN_PAIRS)
    {
        GLOC_DEBUG("[opt] too few observations (%d) - skip", N);
        return false;
    }

    GLOC_DEBUG("[opt] X=%d keyframes N=%d observations", X, N);

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
        // ── Ordering: group 1 = ALL pose variables ────────────────────────────
        //
        // This MUST happen before any residual blocks are added. The rel_pose
        // and prior terms reference every keyframe in the working set, including
        // keyframes that have no direct observations (empty pt_pairs). If those
        // blocks are in the Ceres problem but absent from the ordering, DENSE_SCHUR
        // and ITERATIVE_SCHUR fail immediately (Iterations: -2, cost: -1.0).
        for (int i = 0; i < X; ++i)
        {
            ord->AddElementToGroup(omega_kf[i].data(), 1);
            ord->AddElementToGroup(t_kf[i].data(), 1);
        }

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

        // Save the original initial poses so each candidate starts fresh.
        const auto orig_omega = omega_kf;
        const auto orig_rhos = rhos;

        for (int yk = 0; yk < kNumYaw; ++yk)
        {
            // ── Reset to original initial and apply yaw offset ────────────────
            //
            // The lambda captures omega_kf/rhos by reference and Ceres updates
            // them in-place. Reset before each candidate so every solve starts
            // from a clean state with only the yaw rotation applied.
            omega_kf = orig_omega;
            rhos = orig_rhos;

            const Mat3d Rz = Eigen::AngleAxisd(yk * kYawStep, Vec3d::UnitZ())
                                 .toRotationMatrix();
            for (int i = 0; i < X; ++i)
            {
                const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
                const double norm = ov.norm();
                const Mat3d R_orig = Eigen::AngleAxisd(
                                         norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                         .toRotationMatrix();
                const Eigen::AngleAxisd aa_new(Rz * R_orig);
                const Vec3d ov_new = aa_new.axis() * aa_new.angle();
                omega_kf[i] = {ov_new.x(), ov_new.y(), ov_new.z()};
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

            // Skip failed solves (initial_cost == -1.0 means evaluator failure).
            if (init_summary.initial_cost < 0.0)
                continue;

            if (init_summary.final_cost < best_cost)
            {
                best_cost = init_summary.final_cost;
                best_omega = omega_kf; // omega_kf updated in-place by Ceres
                best_t = t_kf;
                best_rhos = rhos;
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

    GLOC_DEBUG("[opt] %s", main_summary.BriefReport().c_str());

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
    GLOC_DEBUG("[opt] inliers=%d/%d (%.1f%%)",
               inliers, N, inlier_ratio * 100.0);

    if (inlier_ratio < GLOC_MIN_INLIER_RATIO)
    {
        GLOC_WARN("[opt] rejected: inlier ratio %.2f < %.2f",
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

        // R_local_body = kf.R_local  (rotation body -> local)
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
        GLOC_WARN("[opt] zero weight - skip");
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

    GLOC_INFO("[opt] SNAPPED T_map_local t=[%.2f %.2f %.2f]",
              T_map_local_t_.x(), T_map_local_t_.y(), T_map_local_t_.z());

    // ── Publish optimized poses and path ─────────────────────────────────────
    // For each keyframe, compute the average rig centre across all valid
    // gloc modules (multiple cameras on the same rig -> average position).
    if (vins_multi::hasGlocOptimizedSubscribers())
    {
        std::vector<geometry_msgs::Pose> opt_poses;
        std::vector<geometry_msgs::Point> opt_path_pts;
        opt_poses.reserve(X);
        opt_path_pts.reserve(X);

        for (int i = 0; i < X; ++i)
        {
            const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
            const double norm = ov.norm();
            const Mat3d R_world_body = Eigen::AngleAxisd(
                                           norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                           .toRotationMatrix();
            const Vec3d t_world_body(t_kf[i][0], t_kf[i][1], t_kf[i][2]);

            // All keyframes have optimized poses - publish regardless of
            // whether they had valid gloc observations (they're still
            // constrained via relative pose and world prior terms).
            const Quat q_world_body(R_world_body);

            geometry_msgs::Pose pose;
            pose.position.x = t_world_body.x();
            pose.position.y = t_world_body.y();
            pose.position.z = t_world_body.z();
            pose.orientation.x = q_world_body.x();
            pose.orientation.y = q_world_body.y();
            pose.orientation.z = q_world_body.z();
            pose.orientation.w = q_world_body.w();
            opt_poses.push_back(pose);

            geometry_msgs::Point pt;
            pt.x = t_world_body.x();
            pt.y = t_world_body.y();
            pt.z = t_world_body.z();
            opt_path_pts.push_back(pt);
        }

        vins_multi::pubGlocOptimized(opt_poses, opt_path_pts);

        // ── Keyframe status markers ───────────────────────────────────────────
        std::vector<std::pair<Eigen::Vector3d, int>> kf_status_vec;
        kf_status_vec.reserve(X);
        for (int i = 0; i < X; ++i)
        {
            const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
            const double norm = ov.norm();
            const Mat3d R_wb = Eigen::AngleAxisd(
                                   norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                   .toRotationMatrix();
            (void)R_wb;
            const Vec3d pos(t_kf[i][0], t_kf[i][1], t_kf[i][2]);

            // Determine status from per_gloc slots
            int status = 0; // not processed
            bool any_done = false;
            for (const auto &slot : working_set[i].per_gloc)
            {
                if (slot.pipeline_done)
                {
                    any_done = true;
                    if (slot.best_train_idx >= 0)
                    {
                        status = 2; // valid match - green
                        break;
                    }
                }
            }
            if (status == 0 && any_done)
                status = 1; // pipeline done but no valid match - red

            kf_status_vec.push_back({pos, status});
        }
        vins_multi::pubGlocKeyframeStatus(kf_status_vec);

        // ── Match lines: query cam -> train cam ───────────────────────────────
        if (vins_multi::pub_gloc_match_lines.getNumSubscribers() > 0)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> match_pairs;

            for (int i = 0; i < X; ++i)
            {
                const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
                const double norm = ov.norm();
                const Mat3d R_world_body = Eigen::AngleAxisd(
                                               norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                               .toRotationMatrix();
                const Vec3d t_world_body(t_kf[i][0], t_kf[i][1], t_kf[i][2]);

                for (std::size_t g = 0; g < working_set[i].per_gloc.size(); ++g)
                {
                    const auto &slot = working_set[i].per_gloc[g];
                    if (!slot.pipeline_done || slot.best_train_idx < 0)
                        continue;

                    // Query camera position in world:
                    // o_query = R_world_body * (-t_cam_body) + t_world_body
                    const Mat3d R_cb = vins_multi::GLOC_CAM_MODULES[g].ric_[0].toRotationMatrix();
                    const Vec3d t_cb = vins_multi::GLOC_CAM_MODULES[g].tic_[0];
                    const Vec3d o_query = R_world_body * (-t_cb) + t_world_body;

                    // Train camera centre in world
                    const std::size_t ti =
                        static_cast<std::size_t>(slot.best_train_idx);
                    const colmap::Image &train_img = map_.images[ti];
                    const Mat3d R_j = train_img.q_c_w.toRotationMatrix();
                    const Vec3d o_train = -(R_j.transpose() * train_img.t_c_w);

                    match_pairs.push_back({o_query, o_train});
                }
            }

            vins_multi::pubGlocMatchLines(match_pairs);
        }
    }

    // Notify registered consumer (e.g. estimator) on the gloc worker thread.
    // The callback must be lightweight - store and return.
    {
        std::lock_guard<std::mutex> lk(cb_mutex_);
        if (callback_)
            callback_(T_map_local_R_, T_map_local_t_);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// YawOnlyParameterization
//
// Constrains the axis-angle omega[3] (encoding R_body_world) to move only
// along world-Z yaw. Pitch and roll are frozen at their initial values,
// which are trusted from IMU integration.
//
// LocalSize  = 1  (scalar yaw perturbation δ in radians)
// GlobalSize = 3  (axis-angle vector)
//
// Plus(omega, δ):
//   R_new = Rz(δ) * AngleAxis(omega)   ← prepend yaw in world frame
//   omega_new = AngleAxis(R_new)
//
// Jacobian (3×1) at δ=0:
//   d/dδ [log(Rz(δ)*R)] |_{δ=0} = R^T * e_z = third row of R_body_world
// ─────────────────────────────────────────────────────────────────────────────
class YawOnlyParameterization : public ceres::LocalParameterization
{
  public:
    bool Plus(const double *omega,
              const double *delta,
              double *omega_new) const override
    {
        const double norm = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);
        Eigen::Matrix3d R_current;
        if (norm > 1e-8)
        {
            Eigen::AngleAxisd aa(norm, Eigen::Vector3d(omega[0], omega[1], omega[2]) / norm);
            R_current = aa.toRotationMatrix();
        }
        else
            R_current = Eigen::Matrix3d::Identity();

        const Eigen::Matrix3d Rz =
            Eigen::AngleAxisd(delta[0], Eigen::Vector3d::UnitZ()).toRotationMatrix();
        const Eigen::AngleAxisd aa_new(Rz * R_current);
        const Eigen::Vector3d ov = aa_new.axis() * aa_new.angle();
        omega_new[0] = ov.x();
        omega_new[1] = ov.y();
        omega_new[2] = ov.z();
        return true;
    }

    bool ComputeJacobian(const double *omega, double *jacobian) const override
    {
        // Column-major 3×1: d(omega)/d(delta)|_{delta=0} = third row of R_body_world
        const double norm = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);
        Eigen::Matrix3d R;
        if (norm > 1e-8)
        {
            Eigen::AngleAxisd aa(norm, Eigen::Vector3d(omega[0], omega[1], omega[2]) / norm);
            R = aa.toRotationMatrix();
        }
        else
            R = Eigen::Matrix3d::Identity();

        jacobian[0] = R(2, 0);
        jacobian[1] = R(2, 1);
        jacobian[2] = R(2, 2);
        return true;
    }

    int GlobalSize() const override
    {
        return 3;
    }
    int LocalSize() const override
    {
        return 1;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// runOptimization_4DOF
//
// Identical to runOptimization_6DOF except each keyframe's rotation is
// constrained to yaw-only via YawOnlyParameterization. Pitch and roll are
// trusted from IMU and held fixed throughout the solve.
// ─────────────────────────────────────────────────────────────────────────────
bool Gloc::runOptimization_4DOF(std::vector<KeyframeGlocState> &working_set)
{
    using Mat3d = Eigen::Matrix3d;
    using Vec3d = Eigen::Vector3d;
    using Quat = Eigen::Quaterniond;

    const int X = static_cast<int>(working_set.size());

    // ── 1. Guard ──────────────────────────────────────────────────────────────
    int valid_slot_count = 0;
    for (const auto &kf : working_set)
        for (const auto &slot : kf.per_gloc)
            if (slot.pipeline_done && slot.best_train_idx >= 0 &&
                !slot.pt_pairs_undistorted.empty())
                ++valid_slot_count;

    const int min_pairs = snapped_ ? GLOC_MIN_PAIRS : GLOC_MIN_PAIRS_FIRST_SNAP;
    if (valid_slot_count < min_pairs)
    {
        GLOC_DEBUG("[opt4] only %d valid slots (need %d%s) - skip",
                   valid_slot_count, min_pairs, snapped_ ? "" : " first-snap");
        return false;
    }

    // ── 2. Init variables ─────────────────────────────────────────────────────
    std::vector<std::array<double, 3>> omega_kf(X);
    std::vector<std::array<double, 3>> t_kf(X);

    bool snapped_local;
    Mat3d R_map_local;
    Vec3d t_map_local;
    {
        std::lock_guard<std::mutex> lk(snap_mutex_);
        snapped_local = snapped_;
        R_map_local = T_map_local_R_;
        t_map_local = T_map_local_t_;
    }

    const Mat3d R_imu_cam = vins_multi::GLOC_CAM_MODULES[0].ric_[0].toRotationMatrix();
    const Vec3d t_imu_cam = vins_multi::GLOC_CAM_MODULES[0].tic_[0];

    for (int i = 0; i < X; ++i)
    {
        const auto &kf = working_set[i];
        if (snapped_local)
        {
            const Mat3d R_local_body = kf.R_local.toRotationMatrix();
            const Mat3d R_world_body = R_map_local * R_local_body;
            const Vec3d t_world_body = R_map_local * kf.P_local + t_map_local;
            const Eigen::AngleAxisd aa(R_world_body.transpose());
            const Vec3d ov = aa.axis() * aa.angle();
            omega_kf[i] = {ov.x(), ov.y(), ov.z()};
            t_kf[i] = {t_world_body.x(), t_world_body.y(), t_world_body.z()};
        }
        else
        {
            const auto &slot0 = kf.per_gloc[0];
            if (slot0.best_train_idx >= 0)
            {
                const colmap::Image &train_img =
                    map_.images[static_cast<size_t>(slot0.best_train_idx)];
                const Mat3d R_cw = train_img.q_c_w.toRotationMatrix();
                const Vec3d t_cw = train_img.t_c_w;
                const Vec3d oj = -(R_cw.transpose() * t_cw);
                const Mat3d R_bw = R_imu_cam * R_cw;
                const Eigen::AngleAxisd aa(R_bw);
                const Vec3d ov = aa.axis() * aa.angle();
                omega_kf[i] = {ov.x(), ov.y(), ov.z()};
                const Vec3d t_body = oj - R_cw.transpose() * R_imu_cam.transpose() * t_imu_cam;
                t_kf[i] = {t_body.x(), t_body.y(), t_body.z()};
            }
            else
            {
                omega_kf[i] = {0.0, 0.0, 0.0};
                t_kf[i] = {0.0, 0.0, 0.0};
            }
        }
    }

    if (!snapped_local)
    {
        Vec3d t_sum = Vec3d::Zero();
        int n_valid = 0;
        for (int i = 0; i < X; ++i)
            if (working_set[i].per_gloc[0].best_train_idx >= 0)
            {
                t_sum += Vec3d(t_kf[i][0], t_kf[i][1], t_kf[i][2]);
                ++n_valid;
            }
        if (n_valid > 0)
        {
            const Vec3d t_mean = t_sum / n_valid;
            for (int i = 0; i < X; ++i)
                if (working_set[i].per_gloc[0].best_train_idx < 0)
                    t_kf[i] = {t_mean.x(), t_mean.y(), t_mean.z()};
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
                slot.multi_pt_pairs_undistorted.empty())
                continue;

            for (int match_k = 0;
                 match_k < static_cast<int>(slot.voted_train_idxs.size()); ++match_k)
            {
                if (match_k >= static_cast<int>(slot.multi_pt_pairs_undistorted.size()) ||
                    slot.multi_pt_pairs_undistorted[match_k].empty())
                    continue;

                const std::size_t ti =
                    static_cast<std::size_t>(slot.voted_train_idxs[match_k]);

                const colmap::Image &train_img = map_.images[ti];
                const colmap::CameraCalib &train_cal = map_.calibs.at(train_img.camera_id);
                const Mat3d R_j = train_img.q_c_w.toRotationMatrix();
                const Vec3d t_j = train_img.t_c_w;
                const Vec3d o_j = -(R_j.transpose() * t_j);
                const Mat3d R_cb = vins_multi::GLOC_CAM_MODULES[g].ric_[0].toRotationMatrix();
                const Vec3d t_cb = vins_multi::GLOC_CAM_MODULES[g].tic_[0];
                const double fx_q = vins_multi::FOCAL_LENGTH;
                const double fy_q = vins_multi::FOCAL_LENGTH;
                const double cx_q = slot.query_feats.image_size.width / 2.0;
                const double cy_q = slot.query_feats.image_size.height / 2.0;

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

                for (const auto &[pq, pt] : slot.multi_pt_pairs_undistorted[match_k])
                {
                    const Vec3d m_t((pt.x() - train_cal.cx) / train_cal.fx,
                                    (pt.y() - train_cal.cy) / train_cal.fy, 1.0);
                    const Vec3d Rtm = R_j.transpose() * m_t;

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
            } // end match_k loop
        }
    }

    const int N = static_cast<int>(flat_obs.size());
    if (N < GLOC_MIN_PAIRS)
    {
        GLOC_DEBUG("[opt4] too few observations (%d) - skip", N);
        return false;
    }
    GLOC_DEBUG("[opt4] X=%d keyframes N=%d observations", X, N);

    // ── Solver options ────────────────────────────────────────────────────────
    ceres::Solver::Options solver_opts;
    solver_opts.minimizer_type = ceres::TRUST_REGION;
    solver_opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_opts.linear_solver_type = (N <= 2000) ? ceres::DENSE_SCHUR : ceres::ITERATIVE_SCHUR;
    solver_opts.preconditioner_type = (N <= 2000) ? ceres::JACOBI : ceres::SCHUR_JACOBI;
    solver_opts.function_tolerance = 1e-8;
    solver_opts.gradient_tolerance = 1e-10;
    solver_opts.parameter_tolerance = 1e-8;
    solver_opts.minimizer_progress_to_stdout = false;
    solver_opts.num_threads = 1;

    ceres::HuberLoss huber_loss(GLOC_HUBER_DELTA);
    ceres::Problem::Options prob_opts;
    prob_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    // ── Problem builder — same as 6DOF but applies YawOnlyParameterization ───
    auto build_problem = [&](ceres::Problem &prob,
                             ceres::ParameterBlockOrdering *ord,
                             bool add_epi,
                             bool add_rel,
                             bool add_prior) {
        for (int i = 0; i < X; ++i)
        {
            ord->AddElementToGroup(omega_kf[i].data(), 1);
            ord->AddElementToGroup(t_kf[i].data(), 1);
            // Yaw-only: pitch and roll frozen, only yaw (world-Z) is optimized.
            prob.AddParameterBlock(omega_kf[i].data(), 3, new YawOnlyParameterization());
        }

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
                ord->AddElementToGroup(&rhos[k], 0);
            }

            if (add_epi && GLOC_W_EPIPOLAR > 0.0)
            {
                auto *cost = new ceres::AutoDiffCostFunction<GlocEpipolarCost, 1, 3, 3>(
                    new GlocEpipolarCost(flat_obs[k].epi));
                auto *scaled = new ceres::ScaledLoss(
                    &huber_loss, GLOC_W_EPIPOLAR, ceres::DO_NOT_TAKE_OWNERSHIP);
                prob.AddResidualBlock(cost, scaled, om, ti);
            }
        }

        if (add_rel && GLOC_W_REL_POSE > 0.0)
        {
            for (int i = 0; i < X; ++i)
            {
                const int k_max = std::min(i + GLOC_REL_POSE_K, X - 1);
                for (int j = i + 1; j <= k_max; ++j)
                {
                    const auto &kf_i = working_set[i];
                    const auto &kf_j = working_set[j];
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

        if (add_prior && GLOC_W_WORLD_PRIOR > 0.0)
        {
            std::lock_guard<std::mutex> lk(snap_mutex_);
            if (!snapped_)
                return;
            for (int i = 0; i < X; ++i)
            {
                const auto &kf = working_set[i];
                const Mat3d R_local_body = kf.R_local.toRotationMatrix();
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
                prob.AddResidualBlock(cost, scaled, omega_kf[i].data(), t_kf[i].data());
            }
        }
    };

    // ── 4. Init pass: yaw candidates (only when not snapped) ─────────────────
    if (!snapped_local)
    {
        constexpr int kNumYaw = 12;
        const double kYawStep = 2.0 * M_PI / kNumYaw;
        double best_cost = std::numeric_limits<double>::max();
        std::vector<std::array<double, 3>> best_omega = omega_kf;
        std::vector<std::array<double, 3>> best_t = t_kf;
        std::vector<double> best_rhos = rhos;
        const auto orig_omega = omega_kf;
        const auto orig_rhos = rhos;

        for (int yk = 0; yk < kNumYaw; ++yk)
        {
            omega_kf = orig_omega;
            rhos = orig_rhos;
            const Mat3d Rz = Eigen::AngleAxisd(yk * kYawStep, Vec3d::UnitZ()).toRotationMatrix();
            for (int i = 0; i < X; ++i)
            {
                const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
                const double norm = ov.norm();
                const Mat3d R_orig = Eigen::AngleAxisd(
                                         norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                         .toRotationMatrix();
                const Eigen::AngleAxisd aa_new(Rz * R_orig);
                const Vec3d ov_new = aa_new.axis() * aa_new.angle();
                omega_kf[i] = {ov_new.x(), ov_new.y(), ov_new.z()};
            }

            ceres::Problem init_prob(prob_opts);
            auto *init_ord = new ceres::ParameterBlockOrdering;
            build_problem(init_prob, init_ord, true, true, false);

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
            if (init_summary.initial_cost < 0.0)
                continue;
            if (init_summary.final_cost < best_cost)
            {
                best_cost = init_summary.final_cost;
                best_omega = omega_kf;
                best_t = t_kf;
                best_rhos = rhos;
            }
        }
        omega_kf = best_omega;
        t_kf = best_t;
        rhos = best_rhos;
    }

    // ── 5. Main solve ─────────────────────────────────────────────────────────
    ceres::Problem main_prob(prob_opts);
    auto *main_ord = new ceres::ParameterBlockOrdering;
    build_problem(main_prob, main_ord, true, true, true);

    ceres::Solver::Options main_opts = solver_opts;
    main_opts.linear_solver_ordering.reset(main_ord);
    main_opts.max_num_iterations = GLOC_MAX_ITERS;

    ceres::Solver::Summary main_summary;
    ceres::Solve(main_opts, &main_prob, &main_summary);
    GLOC_DEBUG("[opt4] %s", main_summary.BriefReport().c_str());

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
    GLOC_DEBUG("[opt4] inliers=%d/%d (%.1f%%)", inliers, N, inlier_ratio * 100.0);
    if (inlier_ratio < GLOC_MIN_INLIER_RATIO)
    {
        GLOC_WARN("[opt4] rejected: inlier ratio %.2f < %.2f", inlier_ratio, GLOC_MIN_INLIER_RATIO);
        return false;
    }

    // ── 7. Compute T_map_local ────────────────────────────────────────────────
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
        const Mat3d R_local_body = kf.R_local.toRotationMatrix();
        // Strip pitch/roll from R_map_local_i before computing t so that
        // t_map_local_i is consistent with the yaw-only rotation.
        const Mat3d R_map_local_i = yawOnlyR(R_world_body * R_local_body.transpose());
        const Vec3d t_map_local_i = Vec3d(t_kf[i][0], t_kf[i][1], t_kf[i][2]) - R_map_local_i * kf.P_local;

        t_acc += w * t_map_local_i;
        w_total += w;

        Quat q_i(R_map_local_i);
        if (w_total > w)
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
        GLOC_WARN("[opt4] zero weight - skip");
        return false;
    }

    // ── 8. Update snapped state ───────────────────────────────────────────────
    // q_mean is already a pure yaw quaternion since R_map_local_i was
    // yaw-stripped before accumulation — no post-hoc stripping needed.
    Quat q_mean(q_acc_w / w_total, q_acc_v.x() / w_total, q_acc_v.y() / w_total, q_acc_v.z() / w_total);
    q_mean.normalize();

    {
        std::lock_guard<std::mutex> lk(snap_mutex_);
        T_map_local_R_ = q_mean.toRotationMatrix();
        T_map_local_t_ = t_acc / w_total;
        snapped_ = true;
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

    GLOC_INFO("[opt4] SNAPPED T_map_local t=[%.2f %.2f %.2f]",
              T_map_local_t_.x(), T_map_local_t_.y(), T_map_local_t_.z());

    // ── Publish (identical to 6DOF) ───────────────────────────────────────────
    if (vins_multi::hasGlocOptimizedSubscribers())
    {
        std::vector<geometry_msgs::Pose> opt_poses;
        std::vector<geometry_msgs::Point> opt_path_pts;
        opt_poses.reserve(X);
        opt_path_pts.reserve(X);

        for (int i = 0; i < X; ++i)
        {
            const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
            const double norm = ov.norm();
            const Mat3d R_world_body = Eigen::AngleAxisd(
                                           norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                           .toRotationMatrix();
            const Vec3d t_world_body(t_kf[i][0], t_kf[i][1], t_kf[i][2]);
            const Quat q_world_body(R_world_body);

            geometry_msgs::Pose pose;
            pose.position.x = t_world_body.x();
            pose.position.y = t_world_body.y();
            pose.position.z = t_world_body.z();
            pose.orientation.x = q_world_body.x();
            pose.orientation.y = q_world_body.y();
            pose.orientation.z = q_world_body.z();
            pose.orientation.w = q_world_body.w();
            opt_poses.push_back(pose);

            geometry_msgs::Point pt;
            pt.x = t_world_body.x();
            pt.y = t_world_body.y();
            pt.z = t_world_body.z();
            opt_path_pts.push_back(pt);
        }
        vins_multi::pubGlocOptimized(opt_poses, opt_path_pts);

        std::vector<std::pair<Eigen::Vector3d, int>> kf_status_vec;
        kf_status_vec.reserve(X);
        for (int i = 0; i < X; ++i)
        {
            const Vec3d pos(t_kf[i][0], t_kf[i][1], t_kf[i][2]);
            int status = 0;
            bool any_done = false;
            for (const auto &slot : working_set[i].per_gloc)
            {
                if (slot.pipeline_done)
                {
                    any_done = true;
                    if (slot.best_train_idx >= 0)
                    {
                        status = 2;
                        break;
                    }
                }
            }
            if (status == 0 && any_done)
                status = 1;
            kf_status_vec.push_back({pos, status});
        }
        vins_multi::pubGlocKeyframeStatus(kf_status_vec);

        if (vins_multi::pub_gloc_match_lines.getNumSubscribers() > 0)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> match_pairs;
            for (int i = 0; i < X; ++i)
            {
                const Eigen::Map<const Vec3d> ov(omega_kf[i].data());
                const double norm = ov.norm();
                const Mat3d R_world_body = Eigen::AngleAxisd(
                                               norm, norm > 1e-8 ? (ov / norm).eval() : Vec3d::UnitZ())
                                               .toRotationMatrix();
                const Vec3d t_world_body(t_kf[i][0], t_kf[i][1], t_kf[i][2]);
                for (std::size_t g = 0; g < working_set[i].per_gloc.size(); ++g)
                {
                    const auto &slot = working_set[i].per_gloc[g];
                    if (!slot.pipeline_done || slot.best_train_idx < 0)
                        continue;
                    const Mat3d R_cb = vins_multi::GLOC_CAM_MODULES[g].ric_[0].toRotationMatrix();
                    const Vec3d t_cb = vins_multi::GLOC_CAM_MODULES[g].tic_[0];
                    const Vec3d o_query = R_world_body * (-t_cb) + t_world_body;
                    const std::size_t ti = static_cast<std::size_t>(slot.best_train_idx);
                    const colmap::Image &train_img = map_.images[ti];
                    const Mat3d R_j = train_img.q_c_w.toRotationMatrix();
                    const Vec3d o_train = -(R_j.transpose() * train_img.t_c_w);
                    match_pairs.push_back({o_query, o_train});
                }
            }
            vins_multi::pubGlocMatchLines(match_pairs);
        }
    }

    {
        std::lock_guard<std::mutex> lk(cb_mutex_);
        if (callback_)
            callback_(T_map_local_R_, T_map_local_t_);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// runOptimization_FixedRel  (shared implementation)
//
// Relative poses between keyframes are fixed (trusted from VINS). The sole
// optimization variables are T_map_local:
//   omega_map[3]  axis-angle of R_map_local  (local → world)
//   t_map[3]      t_map_local
//
// Each keyframe's world pose is derived analytically inside the functors:
//   R_world_body[i] = R_map_local * R_local_body[i]
//   t_world_body[i] = R_map_local * P_local[i] + t_map_local
//
// No GlocRelPoseCost or GlocWorldPriorCost needed — relative poses are
// exact by construction, and the world prior IS T_map_local itself.
// When use_4dof=true, YawOnlyParameterization is applied to omega_map.
// ─────────────────────────────────────────────────────────────────────────────
bool Gloc::runOptimization_FixedRel_6DOF(std::vector<KeyframeGlocState> &working_set)
{
    return runOptimization_FixedRel(working_set, /*use_4dof=*/false);
}

bool Gloc::runOptimization_FixedRel_4DOF(std::vector<KeyframeGlocState> &working_set)
{
    return runOptimization_FixedRel(working_set, /*use_4dof=*/true);
}

bool Gloc::runOptimization_FixedRel(std::vector<KeyframeGlocState> &working_set,
                                    bool use_4dof)
{
    using Mat3d = Eigen::Matrix3d;
    using Vec3d = Eigen::Vector3d;
    using Quat = Eigen::Quaterniond;

    const int X = static_cast<int>(working_set.size());

    // ── 1. Guard ──────────────────────────────────────────────────────────────
    int valid_slot_count = 0;
    for (const auto &kf : working_set)
        for (const auto &slot : kf.per_gloc)
            if (slot.pipeline_done && slot.best_train_idx >= 0 &&
                !slot.pt_pairs_undistorted.empty())
                ++valid_slot_count;

    const int min_pairs = snapped_ ? GLOC_MIN_PAIRS : GLOC_MIN_PAIRS_FIRST_SNAP;
    if (valid_slot_count < min_pairs)
    {
        GLOC_DEBUG("[opt_fr] only %d valid slots (need %d%s) - skip",
                   valid_slot_count, min_pairs, snapped_ ? "" : " first-snap");
        return false;
    }

    // ── 2. Init T_map_local variables ─────────────────────────────────────────
    // omega_map[3]: AA(R_map_local)   t_map[3]: t_map_local
    std::array<double, 3> omega_map;
    std::array<double, 3> t_map;

    bool snapped_local;
    {
        std::lock_guard<std::mutex> lk(snap_mutex_);
        snapped_local = snapped_;
        if (snapped_local)
        {
            // Seed from current T_map_local
            const Eigen::AngleAxisd aa(T_map_local_R_);
            const Vec3d ov = aa.axis() * aa.angle();
            omega_map = {ov.x(), ov.y(), ov.z()};
            t_map = {T_map_local_t_.x(), T_map_local_t_.y(), T_map_local_t_.z()};
        }
    }

    if (!snapped_local)
    {
        // Seed from the first valid keyframe's matched train image.
        // T_map_local maps local → world, so:
        //   R_map_local = R_world_body[i] * R_local_body[i]^T
        //   t_map_local = t_world_body[i] - R_map_local * P_local[i]
        // where R_world_body[i] comes from the matched train image.
        const Mat3d R_imu_cam =
            vins_multi::GLOC_CAM_MODULES[0].ric_[0].toRotationMatrix();
        const Vec3d t_imu_cam = vins_multi::GLOC_CAM_MODULES[0].tic_[0];

        bool seeded = false;
        for (int i = 0; i < X && !seeded; ++i)
        {
            const auto &slot0 = working_set[i].per_gloc[0];
            if (slot0.best_train_idx < 0)
                continue;

            const colmap::Image &img =
                map_.images[static_cast<size_t>(slot0.best_train_idx)];
            const Mat3d R_cw = img.q_c_w.toRotationMatrix();
            const Vec3d t_cw = img.t_c_w;
            const Vec3d oj = -(R_cw.transpose() * t_cw);

            const Mat3d R_bw = R_imu_cam * R_cw; // R_body_world
            const Mat3d R_wb = R_bw.transpose(); // R_world_body
            const Vec3d t_wb = oj - R_cw.transpose() *
                                        R_imu_cam.transpose() * t_imu_cam;

            const Mat3d R_lb = working_set[i].R_local.toRotationMatrix();
            const Vec3d P_l = working_set[i].P_local;

            const Mat3d R_ml = R_wb * R_lb.transpose(); // R_map_local
            const Vec3d t_ml = t_wb - R_ml * P_l;

            const Eigen::AngleAxisd aa(R_ml);
            const Vec3d ov = aa.axis() * aa.angle();
            omega_map = {ov.x(), ov.y(), ov.z()};
            t_map = {t_ml.x(), t_ml.y(), t_ml.z()};
            seeded = true;
        }
        if (!seeded)
        {
            GLOC_DEBUG("[opt_fr] no valid seed - skip");
            return false;
        }
    }

    // ── 3. Flatten observations ───────────────────────────────────────────────
    struct FlatObs
    {
        GlocFixedRelReprojCost rep;
        GlocFixedRelEpipolarCost epi;
        int kf_idx;
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
                slot.multi_pt_pairs_undistorted.empty())
                continue;

            for (int match_k = 0;
                 match_k < static_cast<int>(slot.voted_train_idxs.size()); ++match_k)
            {
                if (match_k >= static_cast<int>(slot.multi_pt_pairs_undistorted.size()) ||
                    slot.multi_pt_pairs_undistorted[match_k].empty())
                    continue;

                const std::size_t ti =
                    static_cast<std::size_t>(slot.voted_train_idxs[match_k]);

                const colmap::Image &train_img = map_.images[ti];
                const colmap::CameraCalib &train_cal = map_.calibs.at(train_img.camera_id);
                const Mat3d R_j = train_img.q_c_w.toRotationMatrix();
                const Vec3d t_j = train_img.t_c_w;
                const Vec3d o_j = -(R_j.transpose() * t_j);
                const Mat3d R_cb = vins_multi::GLOC_CAM_MODULES[g].ric_[0].toRotationMatrix();
                const Vec3d t_cb = vins_multi::GLOC_CAM_MODULES[g].tic_[0];
                const double fx_q = vins_multi::FOCAL_LENGTH;
                const double fy_q = vins_multi::FOCAL_LENGTH;
                const double cx_q = slot.query_feats.image_size.width / 2.0;
                const double cy_q = slot.query_feats.image_size.height / 2.0;

                // Bake in per-keyframe VINS local pose
                const Mat3d R_lb = kf.R_local.toRotationMatrix();
                double R_lb_arr[9], R_j_arr[9], Rcr_arr[9], t_j_arr[3], tcr_arr[3];
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                    {
                        R_lb_arr[r * 3 + c] = R_lb(r, c);
                        R_j_arr[r * 3 + c] = R_j(r, c);
                        Rcr_arr[r * 3 + c] = R_cb(r, c);
                    }
                t_j_arr[0] = t_j.x();
                t_j_arr[1] = t_j.y();
                t_j_arr[2] = t_j.z();
                tcr_arr[0] = t_cb.x();
                tcr_arr[1] = t_cb.y();
                tcr_arr[2] = t_cb.z();

                for (const auto &[pq, pt] : slot.multi_pt_pairs_undistorted[match_k])
                {
                    const Vec3d m_t((pt.x() - train_cal.cx) / train_cal.fx,
                                    (pt.y() - train_cal.cy) / train_cal.fy, 1.0);
                    const Vec3d Rtm = R_j.transpose() * m_t;

                    GlocFixedRelReprojCost rep{};
                    std::memcpy(rep.R_local_body, R_lb_arr, sizeof(R_lb_arr));
                    rep.P_local[0] = kf.P_local.x();
                    rep.P_local[1] = kf.P_local.y();
                    rep.P_local[2] = kf.P_local.z();
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

                    GlocFixedRelEpipolarCost epi{};
                    std::memcpy(epi.R_local_body, R_lb_arr, sizeof(R_lb_arr));
                    epi.P_local[0] = kf.P_local.x();
                    epi.P_local[1] = kf.P_local.y();
                    epi.P_local[2] = kf.P_local.z();
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

                    flat_obs.push_back({rep, epi, i});
                    rhos.push_back(0.1);
                }
            } // end match_k loop
        }
    }

    const int N = static_cast<int>(flat_obs.size());
    if (N < GLOC_MIN_PAIRS)
    {
        GLOC_DEBUG("[opt_fr] too few observations (%d) - skip", N);
        return false;
    }
    GLOC_DEBUG("[opt_fr] X=%d keyframes N=%d observations (fixed-rel, %s)",
               X, N, use_4dof ? "4DOF" : "6DOF");

    // ── Solver options ────────────────────────────────────────────────────────
    ceres::Solver::Options solver_opts;
    solver_opts.minimizer_type = ceres::TRUST_REGION;
    solver_opts.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // Single variable block (6 or 4 DOF) + N rho blocks: always small enough for DENSE_SCHUR
    solver_opts.linear_solver_type = ceres::DENSE_SCHUR;
    solver_opts.preconditioner_type = ceres::JACOBI;
    solver_opts.function_tolerance = 1e-8;
    solver_opts.gradient_tolerance = 1e-10;
    solver_opts.parameter_tolerance = 1e-8;
    solver_opts.minimizer_progress_to_stdout = false;
    solver_opts.num_threads = 1;

    ceres::HuberLoss huber_loss(GLOC_HUBER_DELTA);
    ceres::Problem::Options prob_opts;
    prob_opts.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    auto build_problem = [&](ceres::Problem &prob,
                             ceres::ParameterBlockOrdering *ord,
                             bool add_epi) {
        // One pose block (omega_map), optionally yaw-only
        ord->AddElementToGroup(omega_map.data(), 1);
        ord->AddElementToGroup(t_map.data(), 1);
        if (use_4dof)
            prob.AddParameterBlock(omega_map.data(), 3, new YawOnlyParameterization());

        for (int k = 0; k < N; ++k)
        {
            if (GLOC_W_REPROJ > 0.0)
            {
                auto *cost = new ceres::AutoDiffCostFunction<
                    GlocFixedRelReprojCost, 2, 3, 3, 1>(
                    new GlocFixedRelReprojCost(flat_obs[k].rep));
                auto *scaled = new ceres::ScaledLoss(
                    &huber_loss, GLOC_W_REPROJ, ceres::DO_NOT_TAKE_OWNERSHIP);
                prob.AddResidualBlock(cost, scaled,
                                      omega_map.data(), t_map.data(), &rhos[k]);
                prob.SetParameterLowerBound(&rhos[k], 0, 1.0 / kMaxDepthM);
                ord->AddElementToGroup(&rhos[k], 0);
            }
            if (add_epi && GLOC_W_EPIPOLAR > 0.0)
            {
                auto *cost = new ceres::AutoDiffCostFunction<
                    GlocFixedRelEpipolarCost, 1, 3, 3>(
                    new GlocFixedRelEpipolarCost(flat_obs[k].epi));
                auto *scaled = new ceres::ScaledLoss(
                    &huber_loss, GLOC_W_EPIPOLAR, ceres::DO_NOT_TAKE_OWNERSHIP);
                prob.AddResidualBlock(cost, scaled, omega_map.data(), t_map.data());
            }
        }
    };

    // ── 4. Init pass: yaw candidates (only when not snapped) ─────────────────
    if (!snapped_local)
    {
        constexpr int kNumYaw = 12;
        const double kYawStep = 2.0 * M_PI / kNumYaw;
        double best_cost = std::numeric_limits<double>::max();
        std::array<double, 3> best_omega = omega_map;
        std::array<double, 3> best_t = t_map;
        std::vector<double> best_rhos = rhos;
        const auto orig_omega = omega_map;
        const auto orig_rhos = rhos;

        for (int yk = 0; yk < kNumYaw; ++yk)
        {
            omega_map = orig_omega;
            rhos = orig_rhos;

            const Eigen::Map<const Eigen::Vector3d> ov(omega_map.data());
            const double norm = ov.norm();
            const Mat3d R_orig = Eigen::AngleAxisd(
                                     norm, norm > 1e-8 ? (ov / norm).eval() : Eigen::Vector3d::UnitZ())
                                     .toRotationMatrix();
            const Mat3d Rz = Eigen::AngleAxisd(yk * kYawStep,
                                               Eigen::Vector3d::UnitZ())
                                 .toRotationMatrix();
            const Eigen::AngleAxisd aa_new(Rz * R_orig);
            const Eigen::Vector3d ov_new = aa_new.axis() * aa_new.angle();
            omega_map = {ov_new.x(), ov_new.y(), ov_new.z()};

            ceres::Problem init_prob(prob_opts);
            auto *init_ord = new ceres::ParameterBlockOrdering;
            build_problem(init_prob, init_ord, /*add_epi=*/true);

            ceres::Solver::Options init_opts = solver_opts;
            init_opts.linear_solver_ordering.reset(init_ord);
            init_opts.max_num_iterations = GLOC_INIT_ITERS;
            init_opts.function_tolerance = 1e-3;
            init_opts.parameter_tolerance = 1e-3;
            init_opts.gradient_tolerance = 1e-3;

            ceres::Solver::Summary init_sum;
            ceres::Solve(init_opts, &init_prob, &init_sum);
            if (init_sum.initial_cost < 0.0)
                continue;
            if (init_sum.final_cost < best_cost)
            {
                best_cost = init_sum.final_cost;
                best_omega = omega_map;
                best_t = t_map;
                best_rhos = rhos;
            }
        }
        omega_map = best_omega;
        t_map = best_t;
        rhos = best_rhos;
    }

    // ── 5. Main solve ─────────────────────────────────────────────────────────
    ceres::Problem main_prob(prob_opts);
    auto *main_ord = new ceres::ParameterBlockOrdering;
    build_problem(main_prob, main_ord, /*add_epi=*/true);

    ceres::Solver::Options main_opts = solver_opts;
    main_opts.linear_solver_ordering.reset(main_ord);
    main_opts.max_num_iterations = GLOC_MAX_ITERS;

    ceres::Solver::Summary main_sum;
    ceres::Solve(main_opts, &main_prob, &main_sum);
    GLOC_DEBUG("[opt_fr] %s", main_sum.BriefReport().c_str());

    // ── 6. Inlier check ───────────────────────────────────────────────────────
    int inliers = 0;
    for (int k = 0; k < N; ++k)
    {
        double res[2];
        flat_obs[k].rep(omega_map.data(), t_map.data(), &rhos[k], res);
        const double err = std::sqrt(res[0] * res[0] + res[1] * res[1]);
        if (err < GLOC_INLIER_THRESH_PX)
            ++inliers;
    }
    const double inlier_ratio = static_cast<double>(inliers) / N;
    GLOC_DEBUG("[opt_fr] inliers=%d/%d (%.1f%%)", inliers, N, inlier_ratio * 100.0);
    if (inlier_ratio < GLOC_MIN_INLIER_RATIO)
    {
        GLOC_WARN("[opt_fr] rejected: inlier ratio %.2f < %.2f",
                  inlier_ratio, GLOC_MIN_INLIER_RATIO);
        return false;
    }

    // ── 7. T_map_local is the result directly ─────────────────────────────────
    const Eigen::Map<const Eigen::Vector3d> ov(omega_map.data());
    const double norm = ov.norm();
    const Mat3d R_ml = Eigen::AngleAxisd(
                           norm, norm > 1e-8 ? (ov / norm).eval() : Eigen::Vector3d::UnitZ())
                           .toRotationMatrix();
    const Vec3d t_ml(t_map[0], t_map[1], t_map[2]);

    // ── 8. Update snapped state ───────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(snap_mutex_);
        T_map_local_R_ = use_4dof ? yawOnlyR(R_ml) : R_ml;
        // When stripping pitch/roll, recompute t_ml consistently with the
        // new rotation using the first valid keyframe as anchor so the
        // world position of that keyframe is preserved.
        if (use_4dof)
        {
            const Mat3d R_stripped = T_map_local_R_;
            // Find first keyframe with valid correspondence to use as anchor
            for (int i = 0; i < X; ++i)
            {
                const auto &kf = working_set[i];
                double w = 0.0;
                for (const auto &slot : kf.per_gloc)
                    if (slot.pipeline_done && slot.best_train_idx >= 0)
                        w += static_cast<double>(slot.pt_pairs_undistorted.size());
                if (w < 1.0)
                    continue;
                // t_world_body_i from optimized omega_map/t_map (consistent with R_ml)
                // Recompute: t_map_new = t_world_body_i - R_stripped * P_local_i
                // where t_world_body_i = R_ml * P_local_i + t_ml (original)
                const Vec3d t_world_body_i = R_ml * kf.P_local + t_ml;
                T_map_local_t_ = t_world_body_i - R_stripped * kf.P_local;
                break;
            }
        }
        else
        {
            T_map_local_t_ = t_ml;
        }
        snapped_ = true;
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

    GLOC_INFO("[opt_fr] SNAPPED T_map_local t=[%.2f %.2f %.2f]",
              t_ml.x(), t_ml.y(), t_ml.z());

    // ── Publish ───────────────────────────────────────────────────────────────
    if (vins_multi::hasGlocOptimizedSubscribers())
    {
        std::vector<geometry_msgs::Pose> opt_poses;
        std::vector<geometry_msgs::Point> opt_path_pts;
        opt_poses.reserve(X);
        opt_path_pts.reserve(X);

        for (int i = 0; i < X; ++i)
        {
            const auto &kf = working_set[i];
            const Mat3d R_lb = kf.R_local.toRotationMatrix();
            const Mat3d R_wb = R_ml * R_lb;
            const Vec3d t_wb = R_ml * kf.P_local + t_ml;
            const Quat q_wb(R_wb);

            geometry_msgs::Pose pose;
            pose.position.x = t_wb.x();
            pose.position.y = t_wb.y();
            pose.position.z = t_wb.z();
            pose.orientation.x = q_wb.x();
            pose.orientation.y = q_wb.y();
            pose.orientation.z = q_wb.z();
            pose.orientation.w = q_wb.w();
            opt_poses.push_back(pose);

            geometry_msgs::Point pt;
            pt.x = t_wb.x();
            pt.y = t_wb.y();
            pt.z = t_wb.z();
            opt_path_pts.push_back(pt);
        }
        vins_multi::pubGlocOptimized(opt_poses, opt_path_pts);

        std::vector<std::pair<Eigen::Vector3d, int>> kf_status_vec;
        kf_status_vec.reserve(X);
        for (int i = 0; i < X; ++i)
        {
            const Vec3d t_wb = R_ml * working_set[i].P_local + t_ml;
            int status = 0;
            bool any_done = false;
            for (const auto &slot : working_set[i].per_gloc)
            {
                if (slot.pipeline_done)
                {
                    any_done = true;
                    if (slot.best_train_idx >= 0)
                    {
                        status = 2;
                        break;
                    }
                }
            }
            if (status == 0 && any_done)
                status = 1;
            kf_status_vec.push_back({t_wb, status});
        }
        vins_multi::pubGlocKeyframeStatus(kf_status_vec);

        if (vins_multi::pub_gloc_match_lines.getNumSubscribers() > 0)
        {
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> match_pairs;
            for (int i = 0; i < X; ++i)
            {
                const auto &kf = working_set[i];
                const Mat3d R_wb = R_ml * kf.R_local.toRotationMatrix();
                const Vec3d t_wb = R_ml * kf.P_local + t_ml;
                for (std::size_t g = 0; g < kf.per_gloc.size(); ++g)
                {
                    const auto &slot = kf.per_gloc[g];
                    if (!slot.pipeline_done || slot.best_train_idx < 0)
                        continue;
                    const Mat3d R_cb = vins_multi::GLOC_CAM_MODULES[g].ric_[0].toRotationMatrix();
                    const Vec3d t_cb = vins_multi::GLOC_CAM_MODULES[g].tic_[0];
                    const Vec3d o_query = R_wb * (-t_cb) + t_wb;
                    const std::size_t ti = static_cast<std::size_t>(slot.best_train_idx);
                    const colmap::Image &train_img = map_.images[ti];
                    const Mat3d R_j = train_img.q_c_w.toRotationMatrix();
                    const Vec3d o_train = -(R_j.transpose() * train_img.t_c_w);
                    match_pairs.push_back({o_query, o_train});
                }
            }
            vins_multi::pubGlocMatchLines(match_pairs);
        }
    }

    {
        std::lock_guard<std::mutex> lk(cb_mutex_);
        if (callback_)
            callback_(T_map_local_R_, T_map_local_t_);
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// runOptimization  (dispatcher)
// ─────────────────────────────────────────────────────────────────────────────
bool Gloc::runOptimization(std::vector<KeyframeGlocState> &working_set)
{
    if (GLOC_FIX_REL_POSES)
        return GLOC_USE_4DOF ? runOptimization_FixedRel_4DOF(working_set)
                             : runOptimization_FixedRel_6DOF(working_set);
    else
        return GLOC_USE_4DOF ? runOptimization_4DOF(working_set)
                             : runOptimization_6DOF(working_set);
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
        const fs::path p = fs::path(GLOC_COLMAP_SPARSE_FOLDER) / f;
        if (!fs::exists(p))
        {
            GLOC_ERROR("[checkPaths] missing COLMAP file: %s", p.string().c_str());
            ok = false;
        }
    }

    if (!fs::exists(GLOC_COLMAP_IMG_FOLDER))
    {
        GLOC_ERROR("[checkPaths] image folder does not exist: %s", GLOC_COLMAP_IMG_FOLDER.c_str());
        ok = false;
    }

    if (!fs::exists(GLOC_DBOW3_DATABASE))
    {
        GLOC_ERROR("[checkPaths] DBoW3 database not found: %s", GLOC_DBOW3_DATABASE.c_str());
        ok = false;
    }

    if (!fs::exists(GLOC_DBOW3_VOCAB))
    {
        GLOC_ERROR("[checkPaths] DBoW3 vocabulary not found: %s", GLOC_DBOW3_VOCAB.c_str());
        ok = false;
    }

    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadColmapData
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::loadColmapData()
{
    const std::string &colmap_dir = GLOC_COLMAP_SPARSE_FOLDER;
    const std::string &img_folder = GLOC_COLMAP_IMG_FOLDER;

    GLOC_INFO("[loadColmap] loading from: %s", colmap_dir.c_str());

    try
    {
        auto img_map = colmap::read_images_bin(colmap_dir + "/images.bin");

        if (!colmap::check_image_paths(img_map, img_folder))
            GLOC_WARN("[loadColmap] some map images are missing on disk");

        map_.calibs = colmap::read_cameras_bin(colmap_dir + "/cameras.bin");

        // Load world transform: use explicit file path if given,
        // otherwise look for world_transform.txt inside colmap_dir.
        Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity();
        if (!GLOC_WORLD_TMAT_FILE.empty())
        {
            // Read the 4×4 matrix directly from the specified file.
            std::ifstream f(GLOC_WORLD_TMAT_FILE);
            if (!f)
            {
                GLOC_WARN("[loadColmap] cannot open world transform: %s - using identity",
                          GLOC_WORLD_TMAT_FILE.c_str());
            }
            else
            {
                int row = 0;
                std::string line;
                while (std::getline(f, line) && row < 4)
                {
                    // Strip inline comments (# ...)
                    const auto hash = line.find('#');
                    if (hash != std::string::npos)
                        line = line.substr(0, hash);
                    std::istringstream ss(line);
                    double v;
                    int col = 0;
                    while (ss >> v && col < 4)
                        world_transform(row, col++) = v;
                    if (col > 0)
                        ++row;
                }
                GLOC_INFO("[loadColmap] loaded world transform from: %s", GLOC_WORLD_TMAT_FILE.c_str());
            }
        }
        else
        {
            world_transform = colmap::load_world_transform(colmap_dir);
        }

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

        GLOC_INFO("[loadColmap] images=%zu cameras=%zu rig_slots=%zu",
                  map_.images.size(), map_.calibs.size(), map_.cam_rig_map.size());
    }
    catch (const std::exception &e)
    {
        GLOC_ERROR("[loadColmap] %s", e.what());
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// loadDatabase
// ─────────────────────────────────────────────────────────────────────────────

bool Gloc::loadDatabase()
{
    GLOC_INFO("[loadDB] loading DBoW3 database ...");

    try
    {
        dbow3::load_dbow3_database(GLOC_DBOW3_DATABASE,
                                   GLOC_DBOW3_VOCAB,
                                   map_.db,
                                   map_.feats);

        GLOC_INFO("[loadDB] db_entries=%zu map_feats=%zu", map_.db.size(), map_.feats.size());
    }
    catch (const std::exception &e)
    {
        GLOC_ERROR("[loadDB] %s", e.what());
        return false;
    }

    return true;
}

} // namespace gloc