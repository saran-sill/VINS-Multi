/**
 * colmap_util.cpp
 *
 * Implementation of COLMAP binary file readers and distortion helpers.
 * See colmap_reader.h for the public API.
 */

#include "colmap_util.h"

#include <Eigen/Geometry>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace colmap
{

// ─────────────────────────────────────────────────────────────────────────────
// Internal binary I/O helpers  (translation-unit scope only)
// ─────────────────────────────────────────────────────────────────────────────

namespace
{

template <typename T>
T read_bin(std::ifstream &f)
{
    T v{};
    f.read(reinterpret_cast<char *>(&v), sizeof(T));
    if (!f)
        throw std::runtime_error("Unexpected EOF while reading binary file.");
    return v;
}

std::string read_cstr(std::ifstream &f)
{
    std::string s;
    char c;
    while (f.get(c) && c != '\0')
        s += c;
    return s;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// SensorId::str()
// ─────────────────────────────────────────────────────────────────────────────

std::string SensorId::str() const
{
    switch (type)
    {
    case SensorType::CAMERA:
        return "Camera[" + std::to_string(id) + "]";
    case SensorType::IMU:
        return "IMU[" + std::to_string(id) + "]";
    default:
        return "Invalid[" + std::to_string(id) + "]";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// read_images_bin
// ─────────────────────────────────────────────────────────────────────────────

std::map<image_t, Image> read_images_bin(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open images.bin: " + path);

    std::map<image_t, Image> out;
    const uint64_t n = read_bin<uint64_t>(f);

    for (uint64_t i = 0; i < n; ++i)
    {
        Image img;
        img.image_id = read_bin<image_t>(f);

        // Quaternion stored as w, x, y, z in the binary file.
        // This is the camera-from-COLMAP-world rotation; "world" here means
        // the raw COLMAP reconstruction frame.  Call apply_world_transform()
        // afterwards to rotate into a user-defined world frame.
        const double qw = read_bin<double>(f);
        const double qx = read_bin<double>(f);
        const double qy = read_bin<double>(f);
        const double qz = read_bin<double>(f);
        img.q_c_w = Eigen::Quaterniond(qw, qx, qy, qz).normalized();

        img.t_c_w.x() = read_bin<double>(f);
        img.t_c_w.y() = read_bin<double>(f);
        img.t_c_w.z() = read_bin<double>(f);

        img.camera_id = read_bin<camera_t>(f);
        img.name = read_cstr(f);

        // Skip 2-D point observations (x, y, point3D_id) per keypoint.
        const uint64_t npts = read_bin<uint64_t>(f);
        for (uint64_t p = 0; p < npts; ++p)
        {
            read_bin<double>(f);    // x
            read_bin<double>(f);    // y
            read_bin<point3D_t>(f); // point3D_id
        }

        out[img.image_id] = std::move(img);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// read_cameras_bin
// ─────────────────────────────────────────────────────────────────────────────

CalibMap read_cameras_bin(const std::string &path)
{
    // Number of intrinsic parameters per COLMAP model ID
    // (matches COLMAP source — models 0..15)
    static const int kNumParams[] = {
        3,  // 0  SimplePinhole:           f cx cy
        4,  // 1  Pinhole:                 fx fy cx cy
        4,  // 2  SimpleRadial:            f cx cy k1
        5,  // 3  Radial:                  f cx cy k1 k2
        8,  // 4  OpenCV:                  fx fy cx cy k1 k2 p1 p2
        8,  // 5  OpenCVFisheye:           fx fy cx cy k1 k2 k3 k4
        12, // 6  FullOpenCV:              fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
        5,  // 7  FOV:                     fx fy cx cy omega
        4,  // 8  SimpleRadialFisheye:     f cx cy k1
        5,  // 9  RadialFisheye:           f cx cy k1 k2
        12, // 10 ThinPrismFisheye
        16, // 11 RadTanThinPrismFisheye
        4,  // 12 SimpleDivision:          f cx cy k
        5,  // 13 Division:                fx fy cx cy k
        5,  // 14 SimpleFisheye:           f cx cy k1 k2
        6,  // 15 Fisheye:                 fx fy cx cy k1 k2
    };

    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open cameras.bin: " + path);

    CalibMap out;
    const uint64_t n = read_bin<uint64_t>(f);

    for (uint64_t i = 0; i < n; ++i)
    {
        const uint32_t cam_id = read_bin<uint32_t>(f);
        const int model = read_bin<int>(f);
        const uint64_t width = read_bin<uint64_t>(f);
        const uint64_t height = read_bin<uint64_t>(f);

        const int np = (model >= 0 && model <= 15) ? kNumParams[model] : 0;
        std::vector<double> params(np);
        for (double &p : params)
            p = read_bin<double>(f);

        CameraCalib cal;
        cal.model_id = model;
        cal.width = width;
        cal.height = height;

        // Models with a single focal length (shared fx == fy):
        //   0 SimplePinhole, 2 SimpleRadial, 3 Radial,
        //   8 SimpleRadialFisheye, 9 RadialFisheye, 12 SimpleDivision, 14 SimpleFisheye
        switch (model)
        {
        case 0:
        case 2:
        case 3:
        case 8:
        case 9:
        case 12:
        case 14:
            cal.fx = cal.fy = params[0];
            cal.cx = params[1];
            cal.cy = params[2];
            cal.dist.assign(params.begin() + 3, params.end());
            break;
        default:
            // All other models: fx, fy, cx, cy followed by distortion
            cal.fx = params[0];
            cal.fy = params[1];
            cal.cx = params[2];
            cal.cy = params[3];
            cal.dist.assign(params.begin() + 4, params.end());
            break;
        }

        cal.K << cal.fx, 0.0, cal.cx,
            0.0, cal.fy, cal.cy,
            0.0, 0.0, 1.0;

        out[cam_id] = std::move(cal);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// load_world_transform
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix4d load_world_transform(const std::string &colmap_dir)
{
    const std::string path = colmap_dir + "/world_transform.txt";

    if (!fs::exists(path))
    {
        std::cout << "      world_transform.txt not found — using identity.\n";
        return Eigen::Matrix4d::Identity();
    }

    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open world_transform.txt: " + path);

    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    int row = 0;
    std::string line;

    while (std::getline(f, line) && row < 4)
    {
        // Strip inline comments
        const auto hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);

        // Skip blank lines
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        std::istringstream ss(line);
        for (int col = 0; col < 4; ++col)
        {
            if (!(ss >> M(row, col)))
                throw std::runtime_error(
                    "Bad world_transform.txt: row " + std::to_string(row));
        }
        ++row;
    }

    if (row < 4)
        throw std::runtime_error("world_transform.txt has fewer than 4 data rows.");

    std::cout << "      world_transform.txt loaded.\n";
    return M;
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_world_transform
//
// See header for full derivation.  In short:
//
//   R_c_colmap, t_c_colmap  — pose read from images.bin  (COLMAP frame)
//   R_M, t_M                — rotation/translation block of the 4×4 M matrix
//                             X_world = R_M * X_colmap + t_M
//
//   R_c_world = R_c_colmap * R_M^T
//   t_c_world = t_c_colmap - R_c_colmap * R_M^T * t_M
// ─────────────────────────────────────────────────────────────────────────────

void apply_world_transform(std::vector<Image> &images,
                           const Eigen::Matrix4d &M)
{
    if (M.isIdentity(1e-10))
        return;

    const Eigen::Matrix3d R_M = M.topLeftCorner<3, 3>();
    const Eigen::Vector3d t_M = M.topRightCorner<3, 1>();

    for (Image &img : images)
    {
        const Eigen::Matrix3d R_c_colmap = img.q_c_w.toRotationMatrix();
        const Eigen::Vector3d t_c_colmap = img.t_c_w;

        const Eigen::Matrix3d R_c_world = R_c_colmap * R_M.transpose();
        const Eigen::Vector3d t_c_world = t_c_colmap - R_c_colmap * R_M.transpose() * t_M;

        img.q_c_w = Eigen::Quaterniond(R_c_world).normalized();
        img.t_c_w = t_c_world;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// read_rigs
// ─────────────────────────────────────────────────────────────────────────────

std::map<rig_t, Rig> read_rigs(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open rigs file: " + path);

    std::map<rig_t, Rig> out;
    const uint64_t nr = read_bin<uint64_t>(f);

    for (uint64_t r = 0; r < nr; ++r)
    {
        Rig rig;
        rig.rig_id = read_bin<rig_t>(f);

        const uint32_t num_sensors = read_bin<uint32_t>(f);

        // First sensor entry is the reference sensor (no pose stored for it)
        if (num_sensors > 0)
        {
            rig.ref_sensor.type = static_cast<SensorType>(read_bin<int>(f));
            rig.ref_sensor.id = read_bin<uint32_t>(f);
        }

        // Remaining entries are non-reference sensors with optional extrinsics
        const uint32_t num_non_ref = (num_sensors > 0) ? num_sensors - 1 : 0;
        for (uint32_t j = 0; j < num_non_ref; ++j)
        {
            RigSensor rs;
            rs.sensor_id.type = static_cast<SensorType>(read_bin<int>(f));
            rs.sensor_id.id = read_bin<uint32_t>(f);
            rs.has_pose = (read_bin<uint8_t>(f) != 0);

            if (rs.has_pose)
            {
                const double qw = read_bin<double>(f);
                const double qx = read_bin<double>(f);
                const double qy = read_bin<double>(f);
                const double qz = read_bin<double>(f);
                rs.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();

                rs.translation.x() = read_bin<double>(f);
                rs.translation.y() = read_bin<double>(f);
                rs.translation.z() = read_bin<double>(f);
            }

            rig.sensors.push_back(std::move(rs));
        }

        out[rig.rig_id] = std::move(rig);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_camera_rig_map
// ─────────────────────────────────────────────────────────────────────────────

std::map<uint32_t, CamRigTransform>
build_camera_rig_map(const std::map<rig_t, Rig> &rigs)
{
    std::map<uint32_t, CamRigTransform> out;

    for (const auto &[rid, rig] : rigs)
    {
        // Reference sensor: identity extrinsic (R = I, t = 0)
        if (rig.ref_sensor.type == SensorType::CAMERA)
        {
            CamRigTransform crt;
            crt.rig_id = rid;
            crt.is_ref = true;
            // R_cam_rig and t_cam_rig default to Identity / Zero
            out[rig.ref_sensor.id] = crt;
        }

        // Non-reference sensors with a known extrinsic
        for (const RigSensor &rs : rig.sensors)
        {
            if (rs.sensor_id.type != SensorType::CAMERA || !rs.has_pose)
                continue;

            CamRigTransform crt;
            crt.rig_id = rid;
            crt.is_ref = false;
            crt.R_cam_rig = rs.rotation.toRotationMatrix();
            crt.t_cam_rig = rs.translation;

            out[rs.sensor_id.id] = crt;
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// check_image_paths
// ─────────────────────────────────────────────────────────────────────────────

bool check_image_paths(const std::map<image_t, Image> &images_map,
                       const std::string &image_folder_path)
{
    int missing = 0;

    for (const auto &[id, img] : images_map)
    {
        const fs::path full_path = fs::path(image_folder_path) / img.name;
        if (!fs::exists(full_path))
        {
            std::cerr << "  [WARN] Image not found: " << full_path.string() << "\n";
            ++missing;
        }
    }

    if (missing > 0)
        std::cerr << "  [WARN] " << missing << " / " << images_map.size()
                  << " image(s) not found under: " << image_folder_path << "\n";

    return missing == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// distort_point
//
// Applies the forward distortion model in normalised image coordinates.
// The result (xd, yd) can be projected to pixels as:
//   u = cal.fx * xd + cal.cx
//   v = cal.fy * yd + cal.cy
// ─────────────────────────────────────────────────────────────────────────────

void distort_point(double xn, double yn,
                   const CameraCalib &cal,
                   double &xd, double &yd)
{
    const double r2 = xn * xn + yn * yn;

    // Safe accessor — returns 0 if index is out of range
    const auto d = [&](int i) -> double {
        return (i < static_cast<int>(cal.dist.size())) ? cal.dist[i] : 0.0;
    };

    switch (cal.model_id)
    {
    // ── Pinhole (no distortion) ───────────────────────────────────────────────
    case 0: // SimplePinhole
    case 1: // Pinhole
        xd = xn;
        yd = yn;
        break;

    // ── Radial k1 only ────────────────────────────────────────────────────────
    case 2: // SimpleRadial:          dist = [k1]
    case 8: // SimpleRadialFisheye:   dist = [k1]
    {
        const double s = 1.0 + d(0) * r2;
        xd = xn * s;
        yd = yn * s;
        break;
    }

    // ── Radial k1, k2 ────────────────────────────────────────────────────────
    case 3:  // Radial:         dist = [k1, k2]
    case 9:  // RadialFisheye:  dist = [k1, k2]
    case 14: // SimpleFisheye:  dist = [k1, k2]
    {
        const double s = 1.0 + d(0) * r2 + d(1) * r2 * r2;
        xd = xn * s;
        yd = yn * s;
        break;
    }

    // ── OpenCV (radial + tangential) ──────────────────────────────────────────
    case 4: // OpenCV: dist = [k1, k2, p1, p2]
    {
        const double s = 1.0 + d(0) * r2 + d(1) * r2 * r2;
        const double dx = 2.0 * d(2) * xn * yn + d(3) * (r2 + 2.0 * xn * xn);
        const double dy = d(2) * (r2 + 2.0 * yn * yn) + 2.0 * d(3) * xn * yn;
        xd = xn * s + dx;
        yd = yn * s + dy;
        break;
    }

    // ── Equidistant fisheye (theta-based) ────────────────────────────────────
    case 5:  // OpenCVFisheye: dist = [k1, k2, k3, k4]
    case 15: // Fisheye:       dist = [k1, k2]
    {
        const double r = std::sqrt(r2);
        const double th = std::atan(r);
        const double th2 = th * th;
        const double th_d = th * (1.0 + d(0) * th2 + d(1) * th2 * th2 + d(2) * th2 * th2 * th2 + d(3) * th2 * th2 * th2 * th2);
        const double scale = (r > 1e-8) ? (th_d / r) : 1.0;
        xd = xn * scale;
        yd = yn * scale;
        break;
    }

    // ── Unknown / unimplemented — pass through ────────────────────────────────
    default:
        xd = xn;
        yd = yn;
        break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// undistort_point
//
// Iterative inverse of distort_point.
//
// Starting from the observed (distorted) normalised coordinate, each step
// corrects by the residual between what the forward model would produce and
// what we actually observed:
//
//   x_{n+1} = x_n + (xn_obs - distort(x_n))
//
// This converges quickly (typically < 5 iterations for moderate distortion).
// The default of 20 iterations is conservative and essentially exact for all
// standard COLMAP distortion magnitudes.
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Vector2d undistort_point(const Eigen::Vector2d &pt,
                                const CameraCalib &cal,
                                int iterations)
{
    // Pinhole models have no distortion — return as-is
    if (cal.model_id == 0 || cal.model_id == 1)
        return pt;

    // Back-project pixel → normalised (distorted) coordinates
    const double xn_obs = (pt.x() - cal.cx) / cal.fx;
    const double yn_obs = (pt.y() - cal.cy) / cal.fy;

    // Iteratively find the undistorted normalised coordinate
    double x = xn_obs;
    double y = yn_obs;
    for (int it = 0; it < iterations; ++it)
    {
        double xd, yd;
        distort_point(x, y, cal, xd, yd);
        x += (xn_obs - xd);
        y += (yn_obs - yd);
    }

    // Re-project undistorted normalised coords back to pixel space
    return Eigen::Vector2d(x * cal.fx + cal.cx,
                           y * cal.fy + cal.cy);
}

} // namespace colmap