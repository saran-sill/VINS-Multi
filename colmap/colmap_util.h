#pragma once

/**
 * colmap_util.h
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace colmap
{

// ─────────────────────────────────────────────────────────────────────────────
// COLMAP scalar typedefs
// ─────────────────────────────────────────────────────────────────────────────

using image_t = uint32_t;
using camera_t = uint32_t;
using point3D_t = uint64_t;
using rig_t = uint32_t;

// ─────────────────────────────────────────────────────────────────────────────
// Image  (one entry from images.bin)
//
// q_c_w / t_c_w encode the camera-from-world transform:
//
//   X_cam = q_c_w * X_world + t_c_w
//
// Note: immediately after read_images_bin() the "world" frame is the raw
// COLMAP reconstruction frame.  After apply_world_transform() is called,
// "world" becomes the frame defined by world_transform.txt.
//
// Useful derived quantities:
//
//   // Camera optical centre in world space
//   Eigen::Vector3d p_world = -(img.q_c_w.inverse() * img.t_c_w);
//
//   // Rotation matrix form
//   Eigen::Matrix3d R_c_w = img.q_c_w.toRotationMatrix();
//
//   // World-from-camera (inverse transform)
//   Eigen::Quaterniond q_w_c = img.q_c_w.inverse();
//   Eigen::Vector3d    t_w_c = -(q_w_c * img.t_c_w);
// ─────────────────────────────────────────────────────────────────────────────

struct Image
{
    image_t image_id{};
    camera_t camera_id{};
    std::string name;

    // Rotation:    X_cam = q_c_w * X_world + t_c_w
    Eigen::Quaterniond q_c_w{Eigen::Quaterniond::Identity()};

    // Translation: X_cam = q_c_w * X_world + t_c_w
    Eigen::Vector3d t_c_w{Eigen::Vector3d::Zero()};
};

// ─────────────────────────────────────────────────────────────────────────────
// Camera calibration  (one entry from cameras.bin)
// ─────────────────────────────────────────────────────────────────────────────

struct CameraCalib
{
    int model_id{0};
    uint64_t width{0};
    uint64_t height{0};

    double fx{1.0}, fy{1.0};
    double cx{0.0}, cy{0.0};
    std::vector<double> dist; // distortion params beyond principal-point/focal

    Eigen::Matrix3d K{Eigen::Matrix3d::Identity()};
};

using CalibMap = std::map<camera_t, CameraCalib>;

// ─────────────────────────────────────────────────────────────────────────────
// Rig types
// ─────────────────────────────────────────────────────────────────────────────

enum class SensorType : int
{
    INVALID = -1,
    CAMERA = 0,
    IMU = 1
};

struct SensorId
{
    SensorType type{SensorType::INVALID};
    uint32_t id{};

    std::string str() const;
};

struct RigSensor
{
    SensorId sensor_id;
    bool has_pose{false};

    // X_sensor = rotation * X_rig + translation
    Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
};

struct Rig
{
    rig_t rig_id{};
    SensorId ref_sensor;
    std::vector<RigSensor> sensors; // non-reference sensors only
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-camera extrinsic relative to its rig reference frame.
//
//   X_cam = R_cam_rig * X_rig + t_cam_rig
//
// For the reference camera:  R_cam_rig = I,  t_cam_rig = 0  (is_ref == true).
// ─────────────────────────────────────────────────────────────────────────────

struct CamRigTransform
{
    rig_t rig_id{};
    bool is_ref{true};
    Eigen::Matrix3d R_cam_rig{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t_cam_rig{Eigen::Vector3d::Zero()};
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API — COLMAP binary readers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Parse COLMAP images.bin.
 *
 * Returned poses are in the raw COLMAP reconstruction frame.
 * Call apply_world_transform() afterwards if a world_transform.txt exists.
 *
 * @param path   Full path to images.bin.
 * @returns      map  image_id → Image
 * @throws       std::runtime_error on I/O failure.
 */
std::map<image_t, Image>
read_images_bin(const std::string &path);

/**
 * Parse COLMAP cameras.bin.
 *
 * @param path   Full path to cameras.bin.
 * @returns      map  camera_id → CameraCalib
 * @throws       std::runtime_error on I/O failure.
 */
CalibMap
read_cameras_bin(const std::string &path);

/**
 * Load a 4×4 world transform from <colmap_dir>/world_transform.txt.
 *
 * The file defines how to go from the COLMAP reconstruction frame to a
 * user-defined world frame:
 *
 *   X_world = M * [X_colmap; 1]
 *
 * Returns Identity (and prints a notice) when the file does not exist.
 *
 * @param colmap_dir  Directory that may contain world_transform.txt.
 * @throws            std::runtime_error on a malformed file.
 */
Eigen::Matrix4d
load_world_transform(const std::string &colmap_dir);

/**
 * Re-express all Image poses from the COLMAP frame into the world frame
 * defined by M (as returned by load_world_transform).
 *
 * After this call every img.q_c_w / img.t_c_w satisfies:
 *
 *   X_cam = q_c_w * X_world + t_c_w
 *
 * where X_world is now in the world frame, not the COLMAP frame.
 *
 * Derivation:
 *   X_world  = R_M * X_colmap + t_M          (definition of M)
 *   X_colmap = R_M^T * (X_world - t_M)       (invert)
 *
 *   X_cam = R_c_colmap * X_colmap        + t_c_colmap     (raw COLMAP pose)
 *         = R_c_colmap * R_M^T * X_world + (t_c_colmap - R_c_colmap * R_M^T * t_M)
 *         = R_c_world  * X_world         + t_c_world       (updated pose)
 *
 * No-op when M == Identity.
 *
 * @param images  Vector of Image structs to update in-place.
 * @param M       4×4 world transform as returned by load_world_transform().
 */
void apply_world_transform(std::vector<Image> &images,
                           const Eigen::Matrix4d &M);

/**
 * Parse COLMAP rigs.bin.
 *
 * @param path   Full path to rigs.bin.
 * @returns      map  rig_id → Rig
 * @throws       std::runtime_error on I/O failure.
 */
std::map<rig_t, Rig>
read_rigs(const std::string &path);

/**
 * Flatten a rig map into a per-camera extrinsic lookup.
 *
 * Only CAMERA-type sensors are included.  The reference camera of each rig
 * gets an identity extrinsic (R = I, t = 0, is_ref = true).
 *
 * @param rigs   As returned by read_rigs().
 * @returns      map  camera_id → CamRigTransform
 */
std::map<uint32_t, CamRigTransform>
build_camera_rig_map(const std::map<rig_t, Rig> &rigs);

// ─────────────────────────────────────────────────────────────────────────────
// Public API — Path validation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check that every image in images_map can be found on disk under image_folder_path.
 *
 * For each Image, the expected path is:
 *   image_folder_path / img.name
 *
 * e.g.  image_folder_path = "/data/train"
 *       img.name          = "cam0/frame_000001.jpg"
 *       expected on disk  : "/data/train/cam0/frame_000001.jpg"
 *
 * Missing files are printed as warnings.
 *
 * @param images_map        As returned by read_images_bin().
 * @param image_folder_path Root folder that should contain the images.
 * @returns                 true if all images exist, false if any are missing.
 */
bool check_image_paths(const std::map<image_t, Image> &images_map,
                       const std::string              &image_folder_path);

// ─────────────────────────────────────────────────────────────────────────────
// Public API — Distortion helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Apply lens distortion to a normalised (undistorted) image point.
 *
 * Input/output are in normalised coordinates:
 *   xn = (px - cx) / fx,   yn = (py - cy) / fy
 *
 * Supported model_id values and their dist[] layout (read from cal):
 *
 *   0, 1     — no distortion (pinhole);  xd = xn, yd = yn
 *   2, 8     — SimpleRadial / SimpleRadialFisheye:      k1
 *   3, 9, 14 — Radial / RadialFisheye / SimpleFisheye:  k1, k2
 *   4        — OpenCV:                                  k1, k2, p1, p2
 *   5, 15    — OpenCVFisheye / Fisheye (equidistant):   k1, k2, k3, k4
 *   other    — treated as no distortion
 *
 * @param xn   Normalised x coordinate (input).
 * @param yn   Normalised y coordinate (input).
 * @param cal  Camera calibration (model_id and dist are read from here).
 * @param xd   Distorted normalised x coordinate (output).
 * @param yd   Distorted normalised y coordinate (output).
 */
void distort_point(double xn, double yn,
                   const CameraCalib &cal,
                   double &xd, double &yd);

/**
 * Remove lens distortion from a pixel coordinate using iterative refinement.
 *
 * The algorithm back-projects the pixel to normalised coordinates, then
 * iteratively corrects the distortion residual until convergence (fixed
 * number of Newton-style steps).
 *
 * For pinhole models (model_id 0 or 1) the input is returned unchanged.
 *
 * Usage:
 *   Eigen::Vector2d undistorted = colmap::undistort_point(distorted, cal);
 *   // undistorted is still in pixel coordinates
 *
 * @param pt          Distorted pixel coordinate (x, y).
 * @param cal         Camera calibration from read_cameras_bin().
 * @param iterations  Number of refinement iterations (default 20).
 * @returns           Undistorted pixel coordinate.
 */
Eigen::Vector2d undistort_point(const Eigen::Vector2d &pt,
                                const CameraCalib &cal,
                                int iterations = 20);



} // namespace colmap