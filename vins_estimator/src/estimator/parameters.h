/*******************************************************
 * Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../utility/utility.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <vector>

using namespace std;

namespace vins_multi
{

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_POSE_EFF = 6,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1,
    SIZE_TD = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

const double FOCAL_LENGTH = 530.0; // 460.0;
const double FRAME_PRIORITY_CONST = 20.0;

#define UNIT_SPHERE_ERROR

struct camera_module_info
{
    int module_id_;
    bool depth_;
    bool stereo_;
    // Whether this module participates in VINS feature tracking and back-end
    // optimization. Only modules with use_for_vins_==true are placed in
    // CAM_MODULES; the VINS estimator iterates CAM_MODULES directly.
    bool use_for_vins_ = false;
    // Whether this module's left (cam0) image is used as a query source for
    // global localization (DBoW3). Only modules with use_for_gloc_==true are
    // placed in GLOC_CAM_MODULES; Gloc iterates GLOC_CAM_MODULES directly.
    // A camera may be in both lists with independent unique_ids per list.
    bool use_for_gloc_ = false;
    // Per-gloc-module temporal-match tolerance (seconds). When gloc resolves
    // a keyframe at time t_kf against this module's ring buffer, it accepts
    // any image whose timestamp is within ±image_match_tol_s_ of t_kf.
    // Used only when use_for_gloc_ is true.
    //
    // Default 0.05 s — slightly more than half a 15 Hz frame interval
    // (66.7 ms / 2 = 33.3 ms), with a small margin to avoid borderline
    // floating-point misses on a perfectly mid-frame query. Tight enough
    // that the body-motion error during the gap is well below DBoW3+ORB
    // noise. Override per-camera in YAML for modules with very different
    // frame rates (e.g. a 5 Hz secondary camera may want 0.1 s).
    double image_match_tol_s_ = 0.05;
    // Per-gloc-module image ring-buffer capacity (slots). Gloc allocates
    // exactly this many slots for the module's image ring; the oldest is
    // evicted when full. Used only when use_for_gloc_ is true.
    //
    // Default 10 — eager resolution drains buffers as soon as a keyframe
    // resolves, so most of the time only a couple of slots are occupied.
    // Increase if your gloc camera arrives noticeably later than VINS
    // keyframes; each extra slot covers one more frame-interval of skew.
    // For a slow camera (e.g. 5 Hz), 5 is usually enough — at 5 Hz a 10-slot
    // buffer spans 2 s, which is more than the VINS window depth.
    int image_ring_buffer_capacity_ = 10;
    int img_width_;
    int img_height_;
    double td_;
    double tr_;
    vector<std::string> img_topic_;
    vector<std::string> calib_file_;
    vector<Eigen::Map<Eigen::Quaterniond>> ric_;
    vector<Eigen::Map<Eigen::Vector3d>> tic_;
    int num_downsamples_ = 0;

    vector<double *> para_Ex_Pose_;

    // void mat2vec(){
    //     for(unsigned int i = 0; i < ric_.size(); i++){
    //         para_Ex_Pose_[i][0] = tic_[i].x();
    //         para_Ex_Pose_[i][1] = tic_[i].y();
    //         para_Ex_Pose_[i][2] = tic_[i].z();

    //         Map<MatrixXd>( &para_Ex_Pose_[i][0], 3, 1) = tic_[0];
    //         Map<MatrixXd>( &para_Ex_Pose_[i][3], 4, 1) = ric_[0].coeffs();

    //         Quaterniond q(ric_[i]);

    //         para_Ex_Pose_[i][3] = q.x();
    //         para_Ex_Pose_[i][4] = q.y();
    //         para_Ex_Pose_[i][5] = q.z();
    //         para_Ex_Pose_[i][6] = q.w();

    //     }

    // }

    // void vec2mat(){

    //     for(unsigned int i = 0; i < ric_.size(); i++){
    //         tic_[i].x() = para_Ex_Pose_[i][0];
    //         tic_[i].y() = para_Ex_Pose_[i][1];
    //         tic_[i].z() = para_Ex_Pose_[i][2];

    //         ric_[i] = Quaterniond(para_Ex_Pose_[i][6],
    //                              para_Ex_Pose_[i][3],
    //                              para_Ex_Pose_[i][4],
    //                              para_Ex_Pose_[i][5]).normalized().toRotationMatrix();
    //     }

    // }

    void set_size()
    {
        if (stereo_)
        {
            img_topic_.resize(2);
            calib_file_.resize(2);

            para_Ex_Pose_.clear();
            para_Ex_Pose_.emplace_back(new double[SIZE_POSE]);
            para_Ex_Pose_.emplace_back(new double[SIZE_POSE]);

            ric_.clear();
            ric_.emplace_back(Eigen::Map<Eigen::Quaterniond>(&para_Ex_Pose_[0][3]));
            ric_.emplace_back(Eigen::Map<Eigen::Quaterniond>(&para_Ex_Pose_[1][3]));

            tic_.clear();
            tic_.emplace_back(Eigen::Map<Eigen::Vector3d>(&para_Ex_Pose_[0][0]));
            tic_.emplace_back(Eigen::Map<Eigen::Vector3d>(&para_Ex_Pose_[1][0]));
        }
        else
        {
            if (depth_)
            {
                img_topic_.resize(2);
                calib_file_.resize(2);
            }
            else
            {
                img_topic_.resize(1);
                calib_file_.resize(1);
            }

            para_Ex_Pose_.resize(1, new double[SIZE_POSE]);

            ric_.clear();
            ric_.emplace_back(Eigen::Map<Eigen::Quaterniond>(&para_Ex_Pose_[0][3]));

            tic_.clear();
            tic_.emplace_back(Eigen::Map<Eigen::Vector3d>(&para_Ex_Pose_[0][0]));
        }
    }
};

struct imu_info
{
    std::string imu_topic_;
    double acc_n_;
    double gyr_n_;
    double acc_w_;
    double gyr_w_;
    // double g_norm_;
    Eigen::Matrix3d rcenterimu_;
    Eigen::Vector3d tcenterimu_;
};

extern int USE_GPU;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;

extern std::vector<camera_module_info> CAM_MODULES;
// Camera modules used for global localization queries (DBoW3). A camera may
// appear in both CAM_MODULES and GLOC_CAM_MODULES with independent unique_ids.
// Indices into GLOC_CAM_MODULES are the gloc-side unique_id used by Gloc's
// per-module image ring buffer.
extern std::vector<camera_module_info> GLOC_CAM_MODULES;
extern imu_info IMU_MODULE;
extern Eigen::Vector3d G;
extern map<int, Eigen::Vector3d> pts_gt;

extern int WINDOW_SIZE;

extern int CV_NUM_THREADS;

// Minimum number of features that must be tracked per camera module.
// Used as a guaranteed allocation floor when distributing the total
// feature budget across modules, and as the lower bound when computing
// MAX_TRACK_NUM_PER_MODULE.
extern int MIN_TRACK_NUM_PER_MODULE;
extern int MAX_TRACK_NUM_PER_MODULE;

// Minimum time (seconds) that must elapse between consecutive back-end
// optimization runs. Caps the optimizer frequency regardless of frame arrival rate.
extern double MIN_OPT_INTERVAL;

// Minimum time gap (seconds) between consecutive frames accepted from
// a single camera module. Acts as a per-module frame rate cap (e.g. 0.05s = 20Hz).
extern double MIN_FRAME_INTERVAL_PER_MODULE;

// Minimum time gap (seconds) between any two adjacent frames in the
// optimization window. Rejects incoming frames that would be inserted
// too close to an existing neighbor — primarily prevents near-simultaneous
// frames from different camera modules crowding the same window slot.
extern double MIN_FRAME_INTERVAL_FOR_OPT;

// Minimum number of frames a feature must be continuously tracked
// before it is included in optimization. Filters out short-lived
// observations that are too noisy to constrain the estimator reliably.
extern int MIN_TRACK_FRAME_FOR_OPT;

extern std::string DEBUG_LEVEL;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern int ESTIMATE_EXTRINSIC;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
extern std::string FISHEYE_MASK;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern int EQUALIZE;
extern double NEW_FEATURE_RATIO_THRESHOLD;

extern int MAX_IMG_BUF_SIZE;

extern double DEPTH_MIN;
extern double DEPTH_MAX;

extern double CLAHE_CLIP_LIMIT;
extern int CLAHE_GRID_SIZE;

extern double GOOD_FEAT_TO_TRACK_QUALITY;

extern double IMG_FREQ;

extern int OPTFLOW_WIN_SIZE;
extern int OPTFLOW_PYR_LEVELS;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern int VIS_CIRCLE_RADIUS;
extern int VIS_ARROW_THICKNESS;

extern std::mutex GPU_MUTEX;

void readParameters(std::string config_file);

} // namespace vins_multi