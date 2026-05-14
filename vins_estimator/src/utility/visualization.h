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
void broadcastWorldOdomTF(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);

} // namespace vins_multi