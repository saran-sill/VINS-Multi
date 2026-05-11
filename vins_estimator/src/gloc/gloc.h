#pragma once

#include <ceres/ceres.h>
#include <chrono>
#include <condition_variable>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "colmap_util.h"
#include "dbow3_util.h"
#include "parameters.h"
#include "point_features.h"

// Forward-declare Estimator so gloc.h does not pull in all estimator headers.
namespace vins_multi
{
    class Estimator;
};

namespace gloc
{

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
    // Checks that all required paths exist before attempting to load anything.
    //
    // @returns  true  on success — ready to localise.
    //           false on any missing path, file error, or inconsistent data.
    // ─────────────────────────────────────────────────────────────────────────
    bool init();

    // ─────────────────────────────────────────────────────────────────────────
    // start_process_thread
    //
    // Starts the background processing loop.
    // Call after init() succeeds.
    // The loop runs until the destructor signals it to stop.
    // ─────────────────────────────────────────────────────────────────────────
    void start_process_thread();

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

    // ── Process loop ─────────────────────────────────────────────────────────

    // Entry point for process_thread_.
    void processLoop();

    // ── Estimator (non-owning) ───────────────────────────────────────────────

    // Set by setEstimator(). Used in processLoop() to read VIO state.
    vins_multi::Estimator *estimator_ptr_{nullptr};

    // ── Map data ───────────────────────────────────────────────────────

    Map map_;

    // ── Thread state ─────────────────────────────────────────────────────────

    std::thread process_thread_;
    std::mutex process_mutex_;
    std::condition_variable process_cv_;
    bool stop_thread_{false};
};

} // namespace gloc