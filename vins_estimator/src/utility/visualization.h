/*******************************************************
 * Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../estimator/estimator.h"
#include "../estimator/parameters.h"
#include "../gloc/gloc.h"
#include "CameraPoseVisualization.h"
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

namespace vins_multi
{

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Estimator &estimator);

void pubTrackImage(const cv::Mat &imgTrack, const double t, const unsigned int cam_unique_id);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator);

void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const Estimator &estimator);

void pubCameraPose(const Estimator &estimator, const unsigned int unique_id);

void pubPointCloud(const Estimator &estimator, const unsigned int unique_id);

void pubTF(const Estimator &estimator);

void pubKeyframe(const Estimator &estimator);

void pubKeyframes(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

void pubCar(const Estimator &estimator, const std_msgs::Header &header);

// Publish COLMAP map image poses as camera frustum markers and rig path.
// Uses latched publishers — call once after gloc.init() succeeds.
void pubGlocMap(const gloc::Gloc &gloc);

// Broadcast the world → odom TF transform from T_map_local.
// Call whenever T_map_local changes (from the gloc callback).
// Convention: X_world = R * X_odom + t
void broadcastWorldOdomTF(const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                          ros::Time stamp = ros::Time(0));

// Publish optimized keyframe rig poses and path in world frame.
// Called from the gloc worker thread after each successful runOptimization.
//   poses     — one pose per keyframe: position = rig centre, orientation = rig rotation
//   path_pts  — same positions in keyframe order for the path
void pubGlocOptimized(const std::vector<geometry_msgs::Pose> &poses,
                      const std::vector<geometry_msgs::Point> &path_pts);

// Returns true if at least one subscriber is listening on either
// gloc/opt_poses or gloc/opt_path. Use to skip message building entirely.
bool hasGlocOptimizedSubscribers();

// Publish keyframe status spheres in world frame.
//   status 2 = green  (valid gloc match)
//   status 1 = red    (pipeline done, no valid match)
//   status 0 = yellow (not yet processed)
void pubGlocKeyframeStatus(
    const std::vector<std::pair<Eigen::Vector3d, int>> &kf_status_vec);

// Exposed for direct subscriber checks in non-visualization code.
extern ros::Publisher pub_gloc_match_lines;
extern ros::Publisher pub_gloc_vote_lines;
extern ros::Publisher pub_gloc_corr_lines;

// Publish magenta lines from query camera positions to matched train image
// camera centres in world frame. Topic: gloc/match_lines
void pubGlocMatchLines(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &query_train_pairs);
void pubGlocVoteLines(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &query_train_pairs);
void pubGlocCorrLines(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &query_train_pairs);

} // namespace vins_multi