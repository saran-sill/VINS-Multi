/*******************************************************
 * kalibr_to_vinsmulti
 *
 * Converts a Kalibr / OpenVINS camchain-imucam.yaml file
 * into the configuration files used by VINS-Multi:
 *   - one main config YAML (with imu_T_cam matrices)
 *   - one per-camera intrinsic calib YAML (KANNALA_BRANDT
 *     for equidistant fisheye, PINHOLE for radtan)
 *
 * KEY CONVERSION
 *   Kalibr's `T_cam_imu` is INVERTED to VINS-Multi's
 *   `imu_T_cam`, because the two projects use opposite
 *   naming conventions for the same physical extrinsic:
 *     Kalibr:      p_cam = T_cam_imu * p_imu
 *     VINS-Multi:  p_imu = imu_T_cam * p_cam
 *
 * MODULE GROUPING
 *   Kalibr files use `device_name` as a "rig leader" marker.
 *   A camera with a non-empty device_name STARTS a new
 *   module; subsequent cameras with no device_name JOIN
 *   the most recent module. Examples:
 *     cam0: device_name=front, cam1: (none)
 *       -> 1 stereo module "front" with cam0+cam1
 *     cam0: device_name=front, cam1: (none), cam2: device_name=down
 *       -> stereo module "front" + mono module "down"
 *
 *   The first cam in a module becomes "left" / cam0,
 *   the second becomes "right" / cam1. Mono modules emit
 *   `stereo: 0` and skip image1/cam1/imu_T_cam1 fields.
 *
 *   `device_name` is REQUIRED on at least one camera.
 *   The converter refuses to guess rig boundaries.
 *
 * Build:
 *   g++ -std=c++17 -O2 kalibr_to_vinsmulti.cpp \
 *       -I/usr/include/eigen3 \
 *       -o kalibr_to_vinsmulti \
 *       `pkg-config --cflags --libs opencv4`
 *
 * Usage:
 *   ./kalibr_to_vinsmulti <camchain.yaml> <output_dir>
 *      [--imu-topic /viso/imu/data_raw]
 *******************************************************/

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =====================================================================
// Data structures
// =====================================================================

struct CamEntry
{
    std::string key;         // "cam0", "cam1", ...
    std::string device_name; // "front_cam", "" if absent
    std::string rostopic;
    std::string camera_model;     // "pinhole"
    std::string distortion_model; // "equidistant" / "radtan"
    int width = 0;
    int height = 0;
    Eigen::Matrix4d T_cam_imu = Eigen::Matrix4d::Identity();
    std::vector<double> intrinsics; // [fx, fy, cx, cy]
    std::vector<double> distortion; // 4 coeffs
    double timeshift_cam_imu = 0.0;
};

struct Module
{
    std::string name;
    std::vector<size_t> cam_indices; // into the flat camera list
};

// =====================================================================
// Math helpers
// =====================================================================

static Eigen::Matrix4d invertSE3(const Eigen::Matrix4d &T)
{
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Ti.block<3, 3>(0, 0) = R.transpose();
    Ti.block<3, 1>(0, 3) = -R.transpose() * t;
    return Ti;
}

static std::string fmt(double v)
{
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    return oss.str();
}

// =====================================================================
// Parsing the Kalibr YAML
// =====================================================================

static bool readMat4(const cv::FileNode &n, Eigen::Matrix4d &M)
{
    if (n.empty() || !n.isSeq() || n.size() != 4)
        return false;
    for (int r = 0; r < 4; ++r)
    {
        cv::FileNode row = n[r];
        if (!row.isSeq() || row.size() != 4)
            return false;
        for (int c = 0; c < 4; ++c)
        {
            M(r, c) = static_cast<double>(row[c]);
        }
    }
    return true;
}

template <typename T>
static bool readVec(const cv::FileNode &n, std::vector<T> &v)
{
    if (n.empty() || !n.isSeq())
        return false;
    v.clear();
    for (auto it = n.begin(); it != n.end(); ++it)
        v.push_back(static_cast<T>(*it));
    return true;
}

static std::vector<CamEntry> parseKalibr(const std::string &path)
{
    cv::FileStorage fs_in(path, cv::FileStorage::READ);
    if (!fs_in.isOpened())
        throw std::runtime_error("cannot open " + path);

    std::vector<CamEntry> cams;
    for (int i = 0;; ++i)
    {
        std::string key = "cam" + std::to_string(i);
        cv::FileNode node = fs_in[key];
        if (node.empty())
            break;

        CamEntry c;
        c.key = key;
        if (!node["device_name"].empty())
            node["device_name"] >> c.device_name;
        if (!node["rostopic"].empty())
            node["rostopic"] >> c.rostopic;
        if (!node["camera_model"].empty())
            node["camera_model"] >> c.camera_model;
        if (!node["distortion_model"].empty())
            node["distortion_model"] >> c.distortion_model;
        if (!node["timeshift_cam_imu"].empty())
            c.timeshift_cam_imu = static_cast<double>(node["timeshift_cam_imu"]);

        std::vector<int> res;
        readVec(node["resolution"], res);
        if (res.size() == 2)
        {
            c.width = res[0];
            c.height = res[1];
        }

        readVec(node["intrinsics"], c.intrinsics);
        readVec(node["distortion_coeffs"], c.distortion);

        if (!readMat4(node["T_cam_imu"], c.T_cam_imu))
            throw std::runtime_error(key + ": missing or malformed T_cam_imu");

        cams.push_back(std::move(c));
    }
    if (cams.empty())
        throw std::runtime_error("no cameras found in " + path);
    return cams;
}

// =====================================================================
// Module grouping
// =====================================================================

// Group cameras by `device_name`: a non-empty device_name STARTS a new
// module (it's the "rig leader"); subsequent cameras with no device_name
// JOIN the most recent module.
//
// `device_name` is REQUIRED on at least the first camera of each rig.
// Without it the converter has no way to know which cameras belong
// together, and silently guessing pairs would produce a broken config.
static std::vector<Module> groupModules(const std::vector<CamEntry> &cams)
{
    bool any_named = false;
    for (const auto &c : cams)
        if (!c.device_name.empty())
            any_named = true;
    if (!any_named)
        throw std::runtime_error(
            "no `device_name` field found on any camera. Add a device_name "
            "to the first camera of each rig in the Kalibr YAML "
            "(e.g. `device_name: front_cam` on cam0, leave cam1 unnamed) "
            "so the converter can group them into modules.");

    std::vector<Module> mods;
    for (size_t i = 0; i < cams.size(); ++i)
    {
        const auto &c = cams[i];
        if (!c.device_name.empty())
        {
            Module m;
            m.name = c.device_name;
            m.cam_indices.push_back(i);
            mods.push_back(std::move(m));
        }
        else
        {
            if (mods.empty())
                throw std::runtime_error(
                    "cam without device_name precedes any named camera");
            mods.back().cam_indices.push_back(i);
        }
    }
    return mods;
}

// =====================================================================
// Per-camera intrinsic YAML
// =====================================================================

// For stereo modules we tag files with _left / _right; for mono we use
// a plain suffix because there's no second camera to disambiguate.
static std::string camCalibFilename(const std::string &module_name,
                                    bool is_stereo, bool is_left)
{
    if (!is_stereo)
        return module_name + "_camera_calib.yaml";
    return module_name + (is_left ? "_left_camera_calib.yaml"
                                  : "_right_camera_calib.yaml");
}

static std::string camDisplayName(const std::string &module_name,
                                  bool is_stereo, bool is_left)
{
    // Strip a trailing "_cam" so e.g. "front_cam" -> "front" -> "front_left".
    std::string base = module_name;
    if (base.size() >= 4 && base.substr(base.size() - 4) == "_cam")
        base = base.substr(0, base.size() - 4);
    if (!is_stereo)
        return base;
    return base + (is_left ? "_left" : "_right");
}

static void writeIntrinsicYaml(const std::string &outpath,
                               const CamEntry &c,
                               const std::string &cam_name)
{
    std::ofstream f(outpath);
    if (!f)
        throw std::runtime_error("cannot write " + outpath);

    std::string model_type;
    if (c.distortion_model == "equidistant")
    {
        model_type = "KANNALA_BRANDT";
    }
    else if (c.distortion_model == "radtan" || c.distortion_model == "plumb_bob")
    {
        model_type = "PINHOLE";
    }
    else
    {
        throw std::runtime_error("unsupported distortion_model '" + c.distortion_model + "' for " + cam_name);
    }

    if (c.intrinsics.size() != 4)
        throw std::runtime_error(cam_name + ": expected 4 intrinsics");
    if (c.distortion.size() != 4)
        throw std::runtime_error(cam_name + ": expected 4 distortion coeffs");

    f << "%YAML:1.0\n";
    f << "---\n";
    f << "model_type: " << model_type << "\n";
    f << "camera_name: \"" << cam_name << "\"\n";
    f << "image_width: " << c.width << "\n";
    f << "image_height: " << c.height << "\n";

    if (model_type == "KANNALA_BRANDT")
    {
        // camodocal: k2..k5 for the 4 fisheye coeffs, mu/mv/u0/v0 for intrinsics
        f << "projection_parameters:\n";
        f << "   k2: " << fmt(c.distortion[0]) << "\n";
        f << "   k3: " << fmt(c.distortion[1]) << "\n";
        f << "   k4: " << fmt(c.distortion[2]) << "\n";
        f << "   k5: " << fmt(c.distortion[3]) << "\n";
        f << "   mu: " << fmt(c.intrinsics[0]) << "\n";
        f << "   mv: " << fmt(c.intrinsics[1]) << "\n";
        f << "   u0: " << fmt(c.intrinsics[2]) << "\n";
        f << "   v0: " << fmt(c.intrinsics[3]) << "\n";
    }
    else
    { // PINHOLE (radtan)
        // Kalibr radtan order: [k1, k2, p1, p2]
        f << "distortion_parameters:\n";
        f << "   k1: " << fmt(c.distortion[0]) << "\n";
        f << "   k2: " << fmt(c.distortion[1]) << "\n";
        f << "   p1: " << fmt(c.distortion[2]) << "\n";
        f << "   p2: " << fmt(c.distortion[3]) << "\n";
        f << "projection_parameters:\n";
        f << "   fx: " << fmt(c.intrinsics[0]) << "\n";
        f << "   fy: " << fmt(c.intrinsics[1]) << "\n";
        f << "   cx: " << fmt(c.intrinsics[2]) << "\n";
        f << "   cy: " << fmt(c.intrinsics[3]) << "\n";
    }
}

// =====================================================================
// Main VINS-Multi YAML
// =====================================================================

static void writeMatBlock(std::ofstream &f, const std::string &key,
                          const Eigen::Matrix4d &M, const std::string &indent)
{
    f << indent << key << ": !!opencv-matrix\n";
    f << indent << "    rows: 4\n";
    f << indent << "    cols: 4\n";
    f << indent << "    dt: d\n";
    f << indent << "    data: [ ";
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            f << fmt(M(r, c));
            if (!(r == 3 && c == 3))
                f << ", ";
        }
        if (r != 3)
            f << "\n"
              << indent << "            ";
    }
    f << " ]\n";
}

static void writeMainYaml(const std::string &outpath,
                          const std::vector<CamEntry> &cams,
                          const std::vector<Module> &mods,
                          const std::string &imu_topic)
{
    std::ofstream f(outpath);
    if (!f)
        throw std::runtime_error("cannot write " + outpath);

    f << "%YAML:1.0\n\n";
    f << "use_gpu: 0\n";
    f << "output_path: \"/tmp/vins_multi_output\"\n\n";

    // -------- IMU --------
    f << "imu:\n";
    f << "    num: 1\n";
    f << "    topic: \"" << imu_topic << "\"\n";
    f << "    acc_n: 0.2\n";
    f << "    gyr_n: 0.05\n";
    f << "    acc_w: 0.01\n";
    f << "    gyr_w: 0.004\n";
    f << "    g_norm: 9.80\n\n";
    f << "    center_T_imu: !!opencv-matrix\n";
    f << "        rows: 4\n";
    f << "        cols: 4\n";
    f << "        dt: d\n";
    f << "        data: [ 1.0, 0.0, 0.0, 0.0,\n";
    f << "                0.0, 1.0, 0.0, 0.0,\n";
    f << "                0.0, 0.0, 1.0, 0.0,\n";
    f << "                0.0, 0.0, 0.0, 1.0 ]\n\n";

    // -------- camera modules --------
    f << "cam_module:\n";
    f << "    num: " << mods.size() << "\n";
    f << "    modules:\n";

    for (const auto &m : mods)
    {
        if (m.cam_indices.empty())
            continue;
        const CamEntry &cam0 = cams[m.cam_indices[0]];
        const bool stereo = (m.cam_indices.size() >= 2);
        const CamEntry *cam1 = stereo ? &cams[m.cam_indices[1]] : nullptr;

        f << "      - cam_id: " << m.name << "\n";
        f << "        depth: 0\n";
        f << "        stereo: " << (stereo ? 1 : 0) << "\n";
        f << "        image_width: " << cam0.width << "\n";
        f << "        image_height: " << cam0.height << "\n";
        f << "        image0_topic: \"" << cam0.rostopic << "\"\n";
        if (stereo)
            f << "        image1_topic: \"" << cam1->rostopic << "\"\n";

        f << "        cam0_calib: \"" << camCalibFilename(m.name, stereo, true) << "\"\n";
        if (stereo)
            f << "        cam1_calib: \"" << camCalibFilename(m.name, stereo, false) << "\"\n";

        f << "        td: " << fmt(cam0.timeshift_cam_imu) << "\n";
        f << "        rolling_shutter: 0\n";
        f << "        rolling_shutter_tr: 0.03333\n\n";

        // The crucial inversion: Kalibr T_cam_imu  ->  VINS-Multi imu_T_cam
        Eigen::Matrix4d imu_T_cam0 = invertSE3(cam0.T_cam_imu);
        writeMatBlock(f, "imu_T_cam0", imu_T_cam0, "        ");

        if (stereo)
        {
            Eigen::Matrix4d imu_T_cam1 = invertSE3(cam1->T_cam_imu);
            f << "\n";
            writeMatBlock(f, "imu_T_cam1", imu_T_cam1, "        ");
        }
        f << "\n";
    }

    // -------- defaults copied from a working VINS-Multi config --------
    f << "debug_level: \"info\"\n\n";

    f << "estimate_extrinsic: 1\n\n";

    f << "multiple_thread: 0\n\n";

    f << "window_size: 21\n";
    f << "max_cnt: 250\n";
    f << "min_dist: 40\n";
    f << "freq: 15\n";
    f << "depth_min: 0.2\n";
    f << "depth_max: 9.0\n";
    f << "F_threshold: 1.0\n";
    f << "show_track: 1\n";
    f << "flow_back: 1\n";
    f << "equalize: 1\n\n";

    f << "clahe_clip_limit: 3.0\n";
    f << "clahe_grid_size: 3\n\n";

    f << "good_feat_to_track_quality: 0.01\n\n";

    f << "optflow_win_size: 15\n\n";
    f << "optflow_pyr_levels: 3\n\n";

    f << "max_solver_time: 0.06\n";
    f << "max_num_iterations: 12\n";
    f << "keyframe_parallax: 10.0\n\n";

    f << "estimate_td: 0\n\n";

    f << "load_previous_pose_graph: 0\n";
    f << "pose_graph_save_path: \"/tmp/vins_multi_pose_graph/\"\n";
    f << "save_image: 0\n\n";

    f << "vis_circle_radius: 5\n";
    f << "vis_arrow_thickness: 3\n";
}

// =====================================================================
// main
// =====================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0]
                  << " <camchain.yaml> <output_dir>"
                  << " [--imu-topic /viso/imu/data_raw]\n";
        return 1;
    }
    const std::string in_path = argv[1];
    const std::string out_dir = argv[2];
    std::string imu_topic = "/viso/imu/data_raw";

    for (int i = 3; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--imu-topic" && i + 1 < argc)
            imu_topic = argv[++i];
        else
        {
            std::cerr << "unknown arg: " << a << "\n";
            return 1;
        }
    }

    fs::create_directories(out_dir);

    auto cams = parseKalibr(in_path);
    auto mods = groupModules(cams);

    std::cout << "Parsed " << cams.size() << " cameras into "
              << mods.size() << " modules:\n";
    for (const auto &m : mods)
    {
        std::cout << "  " << m.name << " ("
                  << (m.cam_indices.size() >= 2 ? "stereo" : "mono")
                  << "):";
        for (auto i : m.cam_indices)
            std::cout << " " << cams[i].key;
        std::cout << "\n";
    }

    // Per-camera intrinsic files
    for (const auto &m : mods)
    {
        const bool stereo = (m.cam_indices.size() >= 2);
        for (size_t k = 0; k < m.cam_indices.size(); ++k)
        {
            const bool is_left = (k == 0);
            const std::string outpath =
                (fs::path(out_dir) / camCalibFilename(m.name, stereo, is_left)).string();
            const std::string camname = camDisplayName(m.name, stereo, is_left);
            writeIntrinsicYaml(outpath, cams[m.cam_indices[k]], camname);
            std::cout << "  wrote " << outpath << "\n";
        }
    }

    // Main config
    const std::string main_path =
        (fs::path(out_dir) / "vins_multi_config.yaml").string();
    writeMainYaml(main_path, cams, mods, imu_topic);
    std::cout << "wrote " << main_path << "\n";

    return 0;
}