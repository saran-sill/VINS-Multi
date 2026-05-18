/*******************************************************
 * Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"

using namespace ros;
using namespace Eigen;

namespace vins_multi
{

ros::Publisher pub_odometry, pub_latest_odometry, pub_latest_odometry_world;
ros::Publisher pub_gloc_map_frustums, pub_gloc_map_path;
ros::Publisher pub_gloc_opt_poses, pub_gloc_opt_path, pub_gloc_kf_status, pub_gloc_match_lines;
ros::Publisher pub_gloc_vote_lines;
ros::Publisher pub_path;
std::vector<ros::Publisher> pub_point_cloud;
ros::Publisher pub_margin_cloud;
ros::Publisher pub_key_poses;
std::vector<ros::Publisher> pub_camera_pose, pub_latest_camera_pose;
std::vector<ros::Publisher> pub_camera_pose_visual;
nav_msgs::Path path;

ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_path;  // NEW: accumulated path of all keyframes
ros::Publisher pub_keyframe_poses; // NEW: pose array of all keyframes (shows orientation)

// ros::Publisher pub_keyframe_point;
std::vector<ros::Publisher> pub_extrinsic;

std::vector<ros::Publisher> pub_image_track;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

Eigen::Vector3d last_pos = Eigen::Vector3d::Zero();
Eigen::Vector3d last_vel = Eigen::Vector3d::Zero();
Eigen::Vector3d last_omega = Eigen::Vector3d::Zero();
Eigen::Quaterniond last_q = Eigen::Quaterniond::Identity();

const double interpolation_alpha = 0.5;

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("odomimu", 1000);
    pub_latest_odometry_world = n.advertise<nav_msgs::Odometry>("odomimu_world", 1000);

    // Latched — published once after map load, any late subscriber still receives.
    pub_gloc_map_frustums = n.advertise<visualization_msgs::MarkerArray>(
        "gloc/map_frustums", 1, /*latch=*/true);
    pub_gloc_map_path = n.advertise<nav_msgs::Path>(
        "gloc/map_path", 1, /*latch=*/true);
    pub_gloc_opt_poses = n.advertise<geometry_msgs::PoseArray>(
        "gloc/opt_poses", 10);
    pub_gloc_opt_path = n.advertise<nav_msgs::Path>(
        "gloc/opt_path", 10);
    pub_gloc_kf_status = n.advertise<visualization_msgs::MarkerArray>(
        "gloc/kf_status", 10);
    pub_gloc_match_lines = n.advertise<visualization_msgs::Marker>(
        "gloc/match_lines", 10);
    pub_gloc_vote_lines = n.advertise<visualization_msgs::Marker>(
        "gloc/vote_lines", 10);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odomimu_lowhz", 1000);
    // pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    // pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_path = n.advertise<nav_msgs::Path>("keyframe_path", 1000);
    pub_keyframe_poses = n.advertise<geometry_msgs::PoseArray>("keyframe_poses", 1000);
    // pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    for (unsigned int i = 1; i <= CAM_MODULES.size(); i++)
    {

        pub_camera_pose.emplace_back(n.advertise<geometry_msgs::PoseStamped>(std::string("camera_pose_") + std::to_string(i), 1000));
        pub_latest_camera_pose.emplace_back(n.advertise<geometry_msgs::PoseStamped>(std::string("imu_propagate_camera_pose_") + std::to_string(i), 1000));
        pub_camera_pose_visual.emplace_back(n.advertise<visualization_msgs::MarkerArray>(std::string("camera_pose_visual_") + std::to_string(i), 1000));
        pub_extrinsic.emplace_back(n.advertise<nav_msgs::Odometry>(std::string("extrinsic_") + std::to_string(i), 1000));
        pub_image_track.emplace_back(n.advertise<sensor_msgs::Image>(std::string("image_track_") + std::to_string(i), 1000));
        pub_point_cloud.emplace_back(n.advertise<sensor_msgs::PointCloud>(std::string("point_cloud_") + std::to_string(i), 1000));
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

void pubGlocMap(const gloc::Gloc &gloc)
{
    const gloc::Map &map = gloc.getMap();

    if (map.images.empty())
    {
        ROS_WARN("[pubGlocMap] Map has no images — nothing to publish.");
        return;
    }

    const ros::Time now = ros::Time::now();
    const std::string frame_id = "world";

    // Helper: compute rig centre and orientation in world frame for an image.
    //
    // Camera centre in world:   o_cam = -R_cw^T * t_cw
    // Rig centre in world:      o_rig = o_cam - R_cw^T * R_cam_rig^T * t_cam_rig
    // Rig orientation in world: q_w_rig = q_wc * q_cam_rig^{-1}
    //
    // If no cam_rig_map entry exists, treat as identity extrinsic (R=I, t=0)
    // and fall back to camera pose.
    auto rig_pose = [&](const colmap::Image &img)
        -> std::pair<Eigen::Vector3d, Eigen::Quaterniond> {
        const Eigen::Matrix3d R_cw = img.q_c_w.toRotationMatrix();
        const Eigen::Vector3d o_cam = -(R_cw.transpose() * img.t_c_w);
        const Eigen::Quaterniond q_wc(R_cw.transpose());

        auto it = map.cam_rig_map.find(img.camera_id);
        if (it == map.cam_rig_map.end())
            return {o_cam, q_wc};

        const colmap::CamRigTransform &cr = it->second;
        const Eigen::Vector3d o_rig =
            o_cam - R_cw.transpose() * cr.R_cam_rig.transpose() * cr.t_cam_rig;
        const Eigen::Quaterniond q_w_rig =
            q_wc * Eigen::Quaterniond(cr.R_cam_rig).inverse();

        return {o_rig, q_w_rig};
    };

    // ── Camera frustum mesh (single TRIANGLE_LIST marker, latched) ───────────
    // All frustums packed into one marker — avoids RViz per-marker overhead
    // and the 2334-marker array limit.

    constexpr float kDepth = 0.15f;
    constexpr float kHalfW = 0.10f;
    constexpr float kHalfH = 0.07f;

    const Eigen::Vector3d tl_c(-kHalfW, -kHalfH, kDepth);
    const Eigen::Vector3d tr_c(kHalfW, -kHalfH, kDepth);
    const Eigen::Vector3d bl_c(-kHalfW, kHalfH, kDepth);
    const Eigen::Vector3d br_c(kHalfW, kHalfH, kDepth);

    auto toPoint = [](const Eigen::Vector3d &v) -> geometry_msgs::Point {
        geometry_msgs::Point p;
        p.x = v.x();
        p.y = v.y();
        p.z = v.z();
        return p;
    };

    visualization_msgs::Marker m;
    m.header.frame_id = frame_id;
    m.header.stamp = now;
    m.ns = "gloc_map_frustums";
    m.id = 0;
    m.type = visualization_msgs::Marker::TRIANGLE_LIST;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = m.scale.y = m.scale.z = 1.0;
    m.pose.orientation.w = 1.0;
    m.color.r = 1.0f;
    m.color.g = 0.0f;
    m.color.b = 0.0f;
    m.color.a = 1.0f;

    m.points.reserve(map.images.size() * 36); // 6 faces × 2 sides × 3 vertices

    for (const auto &img : map.images)
    {
        const Eigen::Matrix3d R_wc = img.q_c_w.toRotationMatrix().transpose();
        const auto [apex, q_w_rig] = rig_pose(img);

        const Eigen::Vector3d tl = R_wc * tl_c + apex;
        const Eigen::Vector3d tr = R_wc * tr_c + apex;
        const Eigen::Vector3d bl = R_wc * bl_c + apex;
        const Eigen::Vector3d br = R_wc * br_c + apex;

        const auto pa = toPoint(apex);
        const auto ptl = toPoint(tl), ptr = toPoint(tr);
        const auto pbl = toPoint(bl), pbr = toPoint(br);

        // Each face added twice (both winding orders) for double-sided rendering
        // since RViz culls back-faces.
#define FACE(a, b, c)      \
    m.points.push_back(a); \
    m.points.push_back(b); \
    m.points.push_back(c); \
    m.points.push_back(a); \
    m.points.push_back(c); \
    m.points.push_back(b);

        // 4 side faces
        FACE(pa, ptl, ptr)
        FACE(pa, pbr, pbl)
        FACE(pa, pbl, ptl)
        FACE(pa, ptr, pbr)
        // near plane
        FACE(ptl, pbl, pbr)
        FACE(ptl, pbr, ptr)

#undef FACE
    }

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(m);
    pub_gloc_map_frustums.publish(marker_array);

    ROS_INFO("[pubGlocMap] Published %zu frustums (single marker) on gloc/map_frustums",
             map.images.size());

    // ── Rig path (nav_msgs/Path, latched) ─────────────────────────────────────
    // Connect rig centres in ascending image_id order.

    std::vector<const colmap::Image *> sorted_imgs;
    sorted_imgs.reserve(map.images.size());
    for (const auto &img : map.images)
        sorted_imgs.push_back(&img);
    std::sort(sorted_imgs.begin(), sorted_imgs.end(),
              [](const colmap::Image *a, const colmap::Image *b) {
                  return a->image_id < b->image_id;
              });

    // ── Rig path (nav_msgs/Path, latched) ─────────────────────────────────────
    // One pose per unique rig position, sorted by image_id of the representative.
    // Multiple cameras on the same rig share the same rig centre — we keep only
    // one path pose per rig to avoid duplicate/overlapping points.
    //
    // Grouping key: cam_rig_map[camera_id].rig_id if available,
    //               otherwise camera_id (each camera is its own "rig").
    //
    // Representative: lowest image_id within the group (deterministic).

    // Build a map: frame_suffix → representative image (lowest image_id).
    // Images captured at the same time share the same numeric suffix in their
    // name (e.g. "../frame_000007.jpg" → "000007"). One path pose per unique
    // frame suffix = one pose per rig capture.
    //
    // Suffix extraction: strip the extension, then take the last run of digits.
    // Falls back to image_id as string if no digits are found.
    auto frame_suffix = [](const std::string &name) -> std::string {
        // Remove extension
        const std::size_t dot = name.rfind('.');
        const std::string stem = (dot == std::string::npos) ? name : name.substr(0, dot);
        // Find last run of digits
        std::size_t end = stem.size();
        while (end > 0 && !std::isdigit(static_cast<unsigned char>(stem[end - 1])))
            --end;
        std::size_t begin = end;
        while (begin > 0 && std::isdigit(static_cast<unsigned char>(stem[begin - 1])))
            --begin;
        return (begin < end) ? stem.substr(begin, end - begin) : stem;
    };

    std::map<std::string, const colmap::Image *> rig_rep; // suffix → image*

    for (const auto *img : sorted_imgs)                    // sorted ascending by image_id
        rig_rep.try_emplace(frame_suffix(img->name), img); // keeps lowest image_id

    ROS_INFO("[pubGlocMap] cam_rig_map size=%zu, unique frame suffixes=%zu, total images=%zu",
             map.cam_rig_map.size(), rig_rep.size(), map.images.size());

    nav_msgs::Path map_path;
    map_path.header.stamp = now;
    map_path.header.frame_id = frame_id;
    map_path.poses.reserve(rig_rep.size());

    // Sort representatives by image_id for a clean chronological path
    std::vector<const colmap::Image *> rep_imgs;
    rep_imgs.reserve(rig_rep.size());
    for (const auto &[suffix, img] : rig_rep)
        rep_imgs.push_back(img);
    std::sort(rep_imgs.begin(), rep_imgs.end(),
              [](const colmap::Image *a, const colmap::Image *b) {
                  return a->image_id < b->image_id;
              });

    for (const auto *img : rep_imgs)
    {
        const auto [o_rig, q_w_rig] = rig_pose(*img);

        geometry_msgs::PoseStamped ps;
        ps.header = map_path.header;
        ps.pose.position.x = o_rig.x();
        ps.pose.position.y = o_rig.y();
        ps.pose.position.z = o_rig.z();
        ps.pose.orientation.x = q_w_rig.x();
        ps.pose.orientation.y = q_w_rig.y();
        ps.pose.orientation.z = q_w_rig.z();
        ps.pose.orientation.w = q_w_rig.w();
        map_path.poses.push_back(ps);
    }

    pub_gloc_map_path.publish(map_path);

    ROS_INFO("[pubGlocMap] Published map path with %zu poses on gloc/map_path",
             map_path.poses.size());
}

void pubLatestOdometry(const Estimator &estimator)
{

    const double t = estimator.state_hist_.back().t_;

    const Eigen::Vector3d &P = estimator.state_hist_.back().P_lpf_;
    const Eigen::Quaterniond &R = estimator.state_hist_.back().Q_lpf_;
    const Eigen::Vector3d &V = estimator.state_hist_.back().V_lpf_;

    const Eigen::Vector3d &omega = estimator.state_hist_.back().un_gyr_;

    const Eigen::Matrix3d &center_R_imu = estimator.imu_module_.rcenterimu_;
    const Eigen::Vector3d &center_T_imu = estimator.imu_module_.tcenterimu_;

    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "odom";

    Eigen::Vector3d w_T_center, v_center, a_center, omega_center;
    Eigen::Matrix3d w_R_center;

    w_R_center = R * center_R_imu;
    w_T_center = P - w_R_center * center_T_imu;

    w_T_center = Utility::lerp(last_pos, w_T_center, interpolation_alpha);

    odometry.pose.pose.position.x = w_T_center.x();
    odometry.pose.pose.position.y = w_T_center.y();
    odometry.pose.pose.position.z = w_T_center.z();

    Eigen::Quaterniond q_center(w_R_center);

    q_center = last_q.slerp(interpolation_alpha, q_center);

    odometry.pose.pose.orientation.x = q_center.x();
    odometry.pose.pose.orientation.y = q_center.y();
    odometry.pose.pose.orientation.z = q_center.z();
    odometry.pose.pose.orientation.w = q_center.w();

    omega_center = center_R_imu * omega;

    v_center = center_R_imu * V - omega_center.cross(center_T_imu);

    v_center = Utility::lerp(last_vel, v_center, interpolation_alpha);

    odometry.twist.twist.linear.x = v_center.x();
    odometry.twist.twist.linear.y = v_center.y();
    odometry.twist.twist.linear.z = v_center.z();

    omega_center = Utility::lerp(last_omega, omega_center, interpolation_alpha);

    odometry.twist.twist.angular.x = omega_center.x();
    odometry.twist.twist.angular.y = omega_center.y();
    odometry.twist.twist.angular.z = omega_center.z();

    pub_latest_odometry.publish(odometry);

    // ── World→odom TF at IMU rate ─────────────────────────────────────────────
    // Broadcast at every IMU update so RViz can transform odomimu into world
    // without extrapolation errors. Use try_lock to avoid blocking the IMU
    // thread; fall back to the cached last-known transform on contention.
    {
        static Eigen::Matrix3d cached_R = Eigen::Matrix3d::Identity();
        static Eigen::Vector3d cached_t = Eigen::Vector3d::Zero();
        if (estimator.t_map_mutex_.try_lock())
        {
            cached_R = estimator.t_map_local_R_;
            cached_t = estimator.t_map_local_t_;
            estimator.t_map_mutex_.unlock();
        }
        broadcastWorldOdomTF(cached_R, cached_t, odometry.header.stamp);
    }

    // ── World-frame odometry (odomimu_world) ──────────────────────────────────
    // Published only when gloc has snapped a valid T_map_local.
    // Convention:  X_world = T_map_local_R_ * X_local + T_map_local_t_
    {
        std::lock_guard<std::mutex> lk(estimator.t_map_mutex_);
        if (estimator.t_map_snapped_)
        {
            const Eigen::Matrix3d &Rm = estimator.t_map_local_R_;
            const Eigen::Vector3d &tm = estimator.t_map_local_t_;

            nav_msgs::Odometry odom_world = odometry;
            odom_world.header.frame_id = "world";

            const Eigen::Vector3d t_world = Rm * w_T_center + tm;
            const Eigen::Matrix3d R_world = Rm * w_R_center;
            const Eigen::Quaterniond q_world(R_world);

            odom_world.pose.pose.position.x = t_world.x();
            odom_world.pose.pose.position.y = t_world.y();
            odom_world.pose.pose.position.z = t_world.z();
            odom_world.pose.pose.orientation.x = q_world.x();
            odom_world.pose.pose.orientation.y = q_world.y();
            odom_world.pose.pose.orientation.z = q_world.z();
            odom_world.pose.pose.orientation.w = q_world.w();

            pub_latest_odometry_world.publish(odom_world);
        }
    }

    last_pos = w_T_center;
    last_q = q_center;
    last_vel = v_center;
    last_omega = omega_center;

    for (unsigned int i = 0; i < estimator.img_trackers_.size(); i++)
    {

        Vector3d P_cam = P + R * estimator.img_trackers_[i]->cam_info_.tic_[0];
        Quaterniond R_cam = Quaterniond(R * estimator.img_trackers_[i]->cam_info_.ric_[0]);

        geometry_msgs::PoseStamped pose_cam;
        pose_cam.header = odometry.header;
        pose_cam.pose.position.x = P_cam.x();
        pose_cam.pose.position.y = P_cam.y();
        pose_cam.pose.position.z = P_cam.z();
        pose_cam.pose.orientation.x = R_cam.x();
        pose_cam.pose.orientation.y = R_cam.y();
        pose_cam.pose.orientation.z = R_cam.z();
        pose_cam.pose.orientation.w = R_cam.w();

        pub_latest_camera_pose[i].publish(pose_cam);

        cameraposevisual.reset();
        // if(estimator.image_frame_window_.cam_wise_image_frame_ptr_[i].empty()){
        if (false)
        {
            cameraposevisual.publish_clear(pub_camera_pose_visual[i], odometry.header);
        }
        else
        {
            cameraposevisual.add_pose(P_cam, R_cam);
            if (estimator.img_trackers_[i]->cam_info_.stereo_)
            {
                Vector3d P1 = P + R * estimator.img_trackers_[i]->cam_info_.tic_[1];
                Quaterniond R1 = R * estimator.img_trackers_[i]->cam_info_.ric_[1];
                cameraposevisual.add_pose(P1, R1);
            }
            cameraposevisual.publish_by(pub_camera_pose_visual[i], odometry.header);
        }
    }
}

void pubTrackImage(const cv::Mat &imgTrack, const double t, const unsigned int cam_unique_id)
{
    if (pub_image_track[cam_unique_id].getNumSubscribers() > 0)
    {
        std_msgs::Header header;
        header.frame_id = "odom";
        header.stamp = ros::Time(t);
        sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
        pub_image_track[cam_unique_id].publish(imgTrackMsg);
    }
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag_ != Estimator::SolverFlag::NON_LINEAR)
        return;
    // printf("position: %f, %f, %f\r", estimator.Ps_[WINDOW_SIZE].x(), estimator.Ps_[WINDOW_SIZE].y(), estimator.Ps_[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("position: " << estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->T_.transpose());
    // ROS_DEBUG_STREAM("orientation: " << estimator.Vs_[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < estimator.img_trackers_.size(); i++)
        {
            // ROS_DEBUG("calibration result for camera %d", i);

            ROS_DEBUG_STREAM("extirnsic tic: " << estimator.img_trackers_[i]->cam_info_.tic_[0].transpose());
            ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.img_trackers_[i]->cam_info_.ric_[0].toRotationMatrix()).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.img_trackers_[i]->cam_info_.ric_[0].toRotationMatrix();
            eigen_T.block<3, 1>(0, 3) = estimator.img_trackers_[i]->cam_info_.tic_[0];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            fs << std::string("body_T_cam0") + std::to_string(estimator.img_trackers_[i]->cam_info_.module_id_) << cv_T;

            if (estimator.img_trackers_[i]->cam_info_.stereo_)
            {
                Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
                eigen_T.block<3, 3>(0, 0) = estimator.img_trackers_[i]->cam_info_.ric_[1].toRotationMatrix();
                eigen_T.block<3, 1>(0, 3) = estimator.img_trackers_[i]->cam_info_.tic_[1];
                cv::Mat cv_T;
                cv::eigen2cv(eigen_T, cv_T);
                fs << std::string("body_T_cam1") + std::to_string(estimator.img_trackers_[i]->cam_info_.module_id_) << cv_T;
            }
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    // if (ESTIMATE_TD){
    //     for (int i = 0; i < estimator.img_trackers_.size(); i++)
    //     {
    //         ROS_INFO("td %d: %f", i, estimator.img_trackers_[i]->cam_info_.td_);
    //     }
    // }
}

void pubOdometry(const Estimator &estimator)
{
    if (estimator.solver_flag_ == Estimator::SolverFlag::NON_LINEAR)
    {
        auto time_stamp = ros::Time(estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->t_);
        nav_msgs::Odometry odometry;
        odometry.header.stamp = time_stamp;
        odometry.header.frame_id = "odom";
        odometry.child_frame_id = "odom";
        Quaterniond tmp_Q(estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->R_);
        Vector3d tmp_P = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->T_;
        Vector3d tmp_V = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->V_;
        odometry.pose.pose.position.x = tmp_P.x();
        odometry.pose.pose.position.y = tmp_P.y();
        odometry.pose.pose.position.z = tmp_P.z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = tmp_V.x();
        odometry.twist.twist.linear.y = tmp_V.y();
        odometry.twist.twist.linear.z = tmp_V.z();
        pub_odometry.publish(odometry);

        if (pub_path.getNumSubscribers() > 0)
        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = time_stamp;
            pose_stamped.header.frame_id = "odom";
            pose_stamped.pose = odometry.pose.pose;
            path.header.stamp = time_stamp;
            path.header.frame_id = "odom";
            path.poses.push_back(pose_stamped);
            pub_path.publish(path);
        }

        // write result to file
        // ofstream foutC(VINS_RESULT_PATH, ios::app);
        // foutC.setf(ios::fixed, ios::floatfield);
        // foutC.precision(0);
        // foutC << time_stamp.toSec() * 1e9 << ",";
        // foutC.precision(5);
        // foutC << tmp_P.x() << ","
        //       << tmp_P.y() << ","
        //       << tmp_P.z() << ","
        //       << tmp_Q.w() << ","
        //       << tmp_Q.x() << ","
        //       << tmp_Q.y() << ","
        //       << tmp_Q.z() << ","
        //       << tmp_V.x() << ","
        //       << tmp_V.y() << ","
        //       << tmp_V.z() << "," << endl;
        // foutC.close();

        // Eigen::Vector3d tmp_T = estimator.Ps_[WINDOW_SIZE];
        // printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.toSec(), tmp_T.x(), tmp_T.y(), tmp_T.z(),
        //                                                   tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
    }
}

void pubKeyPoses(const Estimator &estimator)
{
    if (estimator.key_poses_.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header.stamp = ros::Time(estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->t_);
    key_poses.header.frame_id = "odom";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    // static int key_poses_id = 0;
    key_poses.id = 0; // key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i < estimator.key_poses_.size(); i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses_[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const unsigned int unique_id)
{
    if ((pub_camera_pose[unique_id].getNumSubscribers() > 0) && (estimator.solver_flag_ == Estimator::SolverFlag::NON_LINEAR))
    {
        auto &frame_ptr = estimator.image_frame_window_.cam_wise_image_frame_ptr_[unique_id].back();

        auto stamp = ros::Time{frame_ptr->t_};
        Vector3d P = frame_ptr->T_ + frame_ptr->R_ * estimator.img_trackers_[unique_id]->cam_info_.tic_[0];
        Quaterniond R = frame_ptr->R_ * estimator.img_trackers_[unique_id]->cam_info_.ric_[0];

        geometry_msgs::PoseStamped odometry;
        odometry.header.stamp = stamp;
        odometry.header.frame_id = "odom";
        odometry.pose.position.x = P.x();
        odometry.pose.position.y = P.y();
        odometry.pose.position.z = P.z();
        odometry.pose.orientation.x = R.x();
        odometry.pose.orientation.y = R.y();
        odometry.pose.orientation.z = R.z();
        odometry.pose.orientation.w = R.w();

        pub_camera_pose[unique_id].publish(odometry);
    }
}

void pubPointCloud(const Estimator &estimator, const unsigned int unique_id)
{
    auto stamp = ros::Time{estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->t_};

    if (pub_point_cloud[unique_id].getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud point_cloud, loop_point_cloud;
        point_cloud.header.stamp = stamp;
        point_cloud.header.frame_id = "odom";

        for (auto &it_per_id : estimator.img_trackers_[unique_id]->f_manager_.feature_)
        {
            int used_num;
            used_num = it_per_id.second.feature_per_frame.size();
            if (used_num < 2)
                continue;
            if (it_per_id.second.solve_flag == FeaturePerId::UNINITIALIZED || it_per_id.second.solve_flag == FeaturePerId::OUTLIER)
                continue;
            int imu_i = it_per_id.second.start_frame;
            Vector3d pts_i = it_per_id.second.feature_per_frame.front().point * it_per_id.second.estimated_depth;
            Vector3d w_pts_i = estimator.image_frame_window_.cam_wise_image_frame_ptr_[unique_id][imu_i]->R_ * (estimator.img_trackers_[unique_id]->cam_info_.ric_[0] * pts_i + estimator.img_trackers_[unique_id]->cam_info_.tic_[0]) + estimator.image_frame_window_.cam_wise_image_frame_ptr_[unique_id][imu_i]->T_;

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            point_cloud.points.push_back(p);
        }

        pub_point_cloud[unique_id].publish(point_cloud);
    }

    // pub margined potin
    if (pub_margin_cloud.getNumSubscribers() > 0)
    {
        sensor_msgs::PointCloud margin_cloud;
        margin_cloud.header.stamp = stamp;
        margin_cloud.header.frame_id = "odom";

        auto &margin_frame_ptr = estimator.image_frame_window_.all_image_frame_ptr_.begin()->second;
        int margin_cam_unique_id = margin_frame_ptr->cam_module_unique_id_;

        for (auto &it_per_id : estimator.img_trackers_[margin_cam_unique_id]->f_manager_.feature_)
        {
            int used_num;
            used_num = it_per_id.second.feature_per_frame.size();
            if (used_num < 2)
                continue;
            // if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
            //         continue;

            if (it_per_id.second.start_frame == 0 && it_per_id.second.feature_per_frame.size() <= 2 && it_per_id.second.solve_flag == FeaturePerId::ESTIMATED)
            {
                int imu_i = it_per_id.second.start_frame;
                Vector3d pts_i = it_per_id.second.feature_per_frame.front().point * it_per_id.second.estimated_depth;
                Vector3d w_pts_i = estimator.image_frame_window_.cam_wise_image_frame_ptr_[margin_cam_unique_id][imu_i]->R_ * (estimator.img_trackers_[margin_cam_unique_id]->cam_info_.ric_[0] * pts_i + estimator.img_trackers_[margin_cam_unique_id]->cam_info_.tic_[0]) + estimator.image_frame_window_.cam_wise_image_frame_ptr_[margin_cam_unique_id][imu_i]->T_;

                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                margin_cloud.points.push_back(p);
            }
        }
        pub_margin_cloud.publish(margin_cloud);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcastWorldOdomTF
//
// Broadcasts the world → odom TF transform from T_map_local.
// Called only when T_map_local changes (from the gloc callback via rosNode),
// not at the full estimator rate.
//
// Convention:  X_world = R * X_odom + t
// ─────────────────────────────────────────────────────────────────────────────

void broadcastWorldOdomTF(const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                          ros::Time stamp)
{
    static tf::TransformBroadcaster br;

    // Use the provided sensor timestamp when available so the TF stays
    // synchronised with bag replay time. Fall back to ros::Time::now()
    // only when no stamp is given (e.g. the pre-snap identity broadcast).
    const ros::Time tf_stamp = (stamp.toSec() > 0.0) ? stamp : ros::Time::now();

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));

    const Eigen::Quaterniond q(R);
    transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));

    br.sendTransform(tf::StampedTransform(
        transform, tf_stamp, "world", "odom"));
}

void pubTF(const Estimator &estimator)
{
    if (estimator.solver_flag_ != Estimator::SolverFlag::NON_LINEAR)
        return;

    auto stamp = ros::Time(estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->t_);
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;

    // Broadcast world → odom TF at every estimator update, stamped with the
    // current sensor time. This keeps the TF tree alive at the full estimator
    // rate regardless of how often gloc fires, avoiding extrapolation errors
    // during bag replay or when gloc solves are slow.
    {
        std::lock_guard<std::mutex> lk(estimator.t_map_mutex_);
        broadcastWorldOdomTF(estimator.t_map_local_R_,
                             estimator.t_map_local_t_,
                             stamp);
    }
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->T_;
    correct_q = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->R_;

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, stamp, "odom", "body"));

    // camera frame
    for (unsigned int i = 0; i < estimator.img_trackers_.size(); i++)
    {
        auto &tic = estimator.img_trackers_[i]->cam_info_.tic_[0];
        auto &ric = estimator.img_trackers_[i]->cam_info_.ric_[0];

        transform.setOrigin(tf::Vector3(tic.x(),
                                        tic.y(),
                                        tic.z()));
        q.setW(ric.w());
        q.setX(ric.x());
        q.setY(ric.y());
        q.setZ(ric.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, stamp, "body", std::string{"camera_"} + std::to_string(i + 1)));

        nav_msgs::Odometry odometry;
        odometry.header.stamp = stamp;
        odometry.header.frame_id = "odom";
        odometry.pose.pose.position.x = tic.x();
        odometry.pose.pose.position.y = tic.y();
        odometry.pose.pose.position.z = tic.z();
        odometry.pose.pose.orientation.x = ric.x();
        odometry.pose.pose.orientation.y = ric.y();
        odometry.pose.pose.orientation.z = ric.z();
        odometry.pose.pose.orientation.w = ric.w();
        pub_extrinsic[i].publish(odometry);
    }
}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag_ == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag_ == estimator.MARGIN_OLD)
    {
        auto &P = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->T_;
        auto &R = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->R_;

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->t_);
        odometry.header.frame_id = "odom";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        // printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose.publish(odometry);
    }
}

void pubKeyframes(const Estimator &estimator)
{
    if (estimator.solver_flag_ != Estimator::SolverFlag::NON_LINEAR)
        return;

    auto stamp = ros::Time(estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->t_);

    // ---- existing single latest-frame odometry (unchanged) ----
    if ((pub_keyframe_pose.getNumSubscribers() > 0) && (estimator.marginalization_flag_ == estimator.MARGIN_OLD))
    {
        auto &P = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->T_;
        auto &R = estimator.image_frame_window_.all_image_frame_ptr_.rbegin()->second->R_;

        nav_msgs::Odometry odometry;
        odometry.header.stamp = stamp;
        odometry.header.frame_id = "odom";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        pub_keyframe_pose.publish(odometry);
    }

    // ---- all in-window keyframes (rebuilt every call) ----
    uint32_t num_sub_kf_poses = pub_keyframe_poses.getNumSubscribers();
    uint32_t num_sub_kf_path = pub_keyframe_path.getNumSubscribers();

    if ((num_sub_kf_poses == 0) && (num_sub_kf_path == 0))
        return;

    nav_msgs::Path keyframe_path;
    geometry_msgs::PoseArray keyframe_poses;
    keyframe_path.header.stamp = stamp;
    keyframe_path.header.frame_id = "odom";
    keyframe_poses.header = keyframe_path.header;

    for (const auto &kv : estimator.image_frame_window_.all_image_frame_ptr_)
    {
        if (!kv.second->is_key_frame_)
            continue;

        const auto &P = kv.second->T_;
        const auto &R = kv.second->R_;

        geometry_msgs::Pose pose;
        pose.position.x = P.x();
        pose.position.y = P.y();
        pose.position.z = P.z();
        pose.orientation.x = R.x();
        pose.orientation.y = R.y();
        pose.orientation.z = R.z();
        pose.orientation.w = R.w();

        if (num_sub_kf_path > 0)
        {
            geometry_msgs::PoseStamped ps;
            ps.header.stamp = ros::Time(kv.second->t_);
            ps.header.frame_id = "odom";
            ps.pose = pose;

            keyframe_path.poses.push_back(ps);
        }

        if (num_sub_kf_poses > 0)
            keyframe_poses.poses.push_back(pose);
    }

    if (num_sub_kf_path > 0)
        pub_keyframe_path.publish(keyframe_path);

    if (num_sub_kf_poses > 0)
        pub_keyframe_poses.publish(keyframe_poses);
}

// ─────────────────────────────────────────────────────────────────────────────
// pubGlocOptimized
//
// Publishes optimized keyframe rig poses and path in world frame.
// Called from the gloc worker thread after each successful runOptimization.
//
// Topics:
//   gloc/opt_poses  — geometry_msgs/PoseArray  (one pose per keyframe)
//   gloc/opt_path   — nav_msgs/Path, cyan       (keyframe rig centres in order)
// ─────────────────────────────────────────────────────────────────────────────

bool hasGlocOptimizedSubscribers()
{
    return pub_gloc_opt_poses.getNumSubscribers() > 0 || pub_gloc_opt_path.getNumSubscribers() > 0 || pub_gloc_kf_status.getNumSubscribers() > 0 || pub_gloc_match_lines.getNumSubscribers() > 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// pubGlocKeyframeStatus
//
// Publishes one sphere marker per keyframe coloured by gloc status:
//   Green  — at least one valid gloc match  (best_train_idx >= 0, pipeline_done)
//   Red    — pipeline done but no valid match (voted out / too few inliers)
//   Yellow — not yet processed               (pipeline_done == false)
//
// Topic: gloc/kf_status  (MarkerArray)
// ─────────────────────────────────────────────────────────────────────────────

void pubGlocKeyframeStatus(
    const std::vector<std::pair<Eigen::Vector3d, int>> &kf_status_vec)
{
    if (pub_gloc_kf_status.getNumSubscribers() == 0)
        return;

    visualization_msgs::MarkerArray ma;
    ma.markers.reserve(kf_status_vec.size() + 1);

    // First add a DELETE_ALL marker to clear stale spheres from last round
    visualization_msgs::Marker del;
    del.header.frame_id = "world";
    del.header.stamp = ros::Time::now();
    del.ns = "gloc_kf_status";
    del.action = visualization_msgs::Marker::DELETEALL;
    ma.markers.push_back(del);

    int id = 0;
    for (const auto &[pos, status] : kf_status_vec)
    {
        visualization_msgs::Marker m;
        m.header.frame_id = "world";
        m.header.stamp = ros::Time::now();
        m.ns = "gloc_kf_status";
        m.id = id++;
        m.type = visualization_msgs::Marker::SPHERE;
        m.action = visualization_msgs::Marker::ADD;

        m.pose.position.x = pos.x();
        m.pose.position.y = pos.y();
        m.pose.position.z = pos.z();
        m.pose.orientation.w = 1.0;

        m.scale.x = m.scale.y = m.scale.z = 0.15;

        // status: 2=valid(green), 1=no match(red), 0=not processed(yellow)
        if (status == 2) // green — valid gloc match
        {
            m.color.r = 0.0f;
            m.color.g = 1.0f;
            m.color.b = 0.0f;
        }
        else if (status == 1) // red — pipeline done, no valid match
        {
            m.color.r = 1.0f;
            m.color.g = 0.0f;
            m.color.b = 0.0f;
        }
        else // yellow — not processed
        {
            m.color.r = 1.0f;
            m.color.g = 1.0f;
            m.color.b = 0.0f;
        }
        m.color.a = 1.0f;

        ma.markers.push_back(m);
    }

    pub_gloc_kf_status.publish(ma);
}

void pubGlocOptimized(const std::vector<geometry_msgs::Pose> &poses,
                      const std::vector<geometry_msgs::Point> &path_pts)
{
    const bool has_pose_sub = pub_gloc_opt_poses.getNumSubscribers() > 0;
    const bool has_path_sub = pub_gloc_opt_path.getNumSubscribers() > 0;

    if (!has_pose_sub && !has_path_sub)
        return;

    const ros::Time now = ros::Time::now();

    // ── PoseArray ─────────────────────────────────────────────────────────────
    if (has_pose_sub)
    {
        geometry_msgs::PoseArray pose_array;
        pose_array.header.stamp = now;
        pose_array.header.frame_id = "world";
        pose_array.poses = poses;
        pub_gloc_opt_poses.publish(pose_array);
    }

    // ── Path (cyan — set in RViz display settings) ────────────────────────────
    if (has_path_sub)
    {
        nav_msgs::Path path;
        path.header.stamp = now;
        path.header.frame_id = "world";
        path.poses.reserve(path_pts.size());

        for (const auto &pt : path_pts)
        {
            geometry_msgs::PoseStamped ps;
            ps.header = path.header;
            ps.pose.position = pt;
            ps.pose.orientation.w = 1.0;
            path.poses.push_back(ps);
        }
        pub_gloc_opt_path.publish(path);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pubGlocMatchLines
//
// Publishes magenta lines connecting each query camera position to its matched
// COLMAP train image camera centre in world frame.
//
// query_train_pairs: vector of (query_pos_world, train_pos_world)
// Topic: gloc/match_lines  (single LINE_LIST marker)
// ─────────────────────────────────────────────────────────────────────────────

void pubGlocMatchLines(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &query_train_pairs)
{
    if (pub_gloc_match_lines.getNumSubscribers() == 0)
        return;

    visualization_msgs::Marker m;
    m.header.frame_id = "world";
    m.header.stamp = ros::Time::now();
    m.ns = "gloc_match_lines";
    m.id = 0;
    m.type = visualization_msgs::Marker::LINE_LIST;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.1;
    m.pose.orientation.w = 1.0;

    // Magenta
    m.color.r = 1.0f;
    m.color.g = 0.0f;
    m.color.b = 1.0f;
    m.color.a = 1.0f;

    m.points.reserve(query_train_pairs.size() * 2);
    for (const auto &[q_pos, t_pos] : query_train_pairs)
    {
        geometry_msgs::Point pq, pt;
        pq.x = q_pos.x();
        pq.y = q_pos.y();
        pq.z = q_pos.z();
        pt.x = t_pos.x();
        pt.y = t_pos.y();
        pt.z = t_pos.z();
        m.points.push_back(pq);
        m.points.push_back(pt);
    }

    pub_gloc_match_lines.publish(m);
}

// ─────────────────────────────────────────────────────────────────────────────
// pubGlocVoteLines
//
// Visualizes consensus voting results after runOrbAndDbow + runConsensusVoting.
// Draws lines in world frame between:
//   - query keyframe local position (used directly as world position)
//   - matched train image camera centre in COLMAP world
// Published on gloc/vote_lines (cyan lines).
// ─────────────────────────────────────────────────────────────────────────────
void pubGlocVoteLines(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &query_train_pairs)
{
    if (pub_gloc_vote_lines.getNumSubscribers() == 0)
        return;

    visualization_msgs::Marker m;
    m.header.frame_id = "world";
    m.header.stamp = ros::Time::now();
    m.ns = "gloc_vote_lines";
    m.id = 0;
    m.type = visualization_msgs::Marker::LINE_LIST;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.025;
    m.pose.orientation.w = 1.0;

    // Cyan
    m.color.r = 1.0f;
    m.color.g = 1.0f;
    m.color.b = 0.0f;
    m.color.a = 1.0f;

    m.points.reserve(query_train_pairs.size() * 2);
    for (const auto &[q_pos, t_pos] : query_train_pairs)
    {
        geometry_msgs::Point pq, pt;
        pq.x = q_pos.x();
        pq.y = q_pos.y();
        pq.z = q_pos.z();
        pt.x = t_pos.x();
        pt.y = t_pos.y();
        pt.z = t_pos.z();
        m.points.push_back(pq);
        m.points.push_back(pt);
    }

    pub_gloc_vote_lines.publish(m);
}

} // namespace vins_multi