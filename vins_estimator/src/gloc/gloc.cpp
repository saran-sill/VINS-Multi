#include "gloc.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace gloc
{

Gloc::Gloc()
{
}

Gloc::~Gloc()
{
    {
        std::lock_guard<std::mutex> lk(process_mutex_);
        stop_thread_ = true;
    }
    process_cv_.notify_all();

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
// processLoop
// ─────────────────────────────────────────────────────────────────────────────

void Gloc::processLoop()
{
    std::cout << "[Gloc] processLoop running.\n";

    while (true)
    {
        std::unique_lock<std::mutex> lk(process_mutex_);

        // Wait until there is work to do or we are asked to stop.
        // TODO: replace the `false` predicate with an actual data-ready check,
        //       e.g. `!input_queue_.empty()`, once the input type is decided.
        process_cv_.wait(lk, [this] {
            return stop_thread_ /* || !input_queue_.empty() */;
        });

        if (stop_thread_)
            break;

        lk.unlock();

        // TODO: pop from input_queue_ and run localisation here.
        // estimator_ptr_ is available to read VIO state.
        // map_ holds all map data (images, calibs, cam_rig_map, db, feats).
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