/*******************************************************
 * Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

namespace vins_multi
{

int USE_GPU = 0;
// std::mutex GPU_MUTEX;

double INIT_DEPTH;
double MIN_PARALLAX;

std::vector<camera_module_info> CAM_MODULES;
imu_info IMU_MODULE;
Eigen::Vector3d G(0.0, 0.0, 9.8);
map<int, Eigen::Vector3d> pts_gt;

std::string DEBUG_LEVEL;

int CV_NUM_THREADS;

int WINDOW_SIZE;

int MIN_TRACK_NUM_PER_MODULE;
double MIN_OPT_INTERVAL;
double MIN_FRAME_INTERVAL_PER_MODULE;
double MIN_FRAME_INTERVAL_FOR_OPT;
int MIN_TRACK_FRAME_FOR_OPT;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int USE_IMU;
int MULTIPLE_THREAD;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
double DEPTH_MIN;
double DEPTH_MAX;

double NEW_FEATURE_RATIO_THRESHOLD;

int MAX_IMG_BUF_SIZE;

double CLAHE_CLIP_LIMIT;
int CLAHE_GRID_SIZE;

double GOOD_FEAT_TO_TRACK_QUALITY;

double IMG_FREQ;

int OPTFLOW_WIN_SIZE;
int OPTFLOW_PYR_LEVELS;

int EQUALIZE;

int MAX_TRACK_NUM_PER_MODULE;

int VIS_CIRCLE_RADIUS;
int VIS_ARROW_THICKNESS;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

#ifdef WITH_CUDA
    USE_GPU = fsSettings["use_gpu"];
#endif
    ROS_WARN("USE_GPU: %d", USE_GPU);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    ESTIMATE_TD = fsSettings["estimate_td"];

    cv::FileNode imu_node = fsSettings["imu"];

    USE_IMU = imu_node["num"];
    ROS_WARN("USE_IMU: %d", USE_IMU);
    if (USE_IMU)
    {
        imu_node["topic"] >> IMU_MODULE.imu_topic_;
        printf("IMU_TOPIC: %s\n", IMU_MODULE.imu_topic_.c_str());

        cv::Mat cv_center_T_imu;
        imu_node["center_T_imu"] >> cv_center_T_imu;
        Eigen::Matrix4d T_temp;
        cv::cv2eigen(cv_center_T_imu, T_temp);
        IMU_MODULE.rcenterimu_ = T_temp.block<3, 3>(0, 0);
        IMU_MODULE.tcenterimu_ = T_temp.block<3, 1>(0, 3);

        IMU_MODULE.acc_n_ = imu_node["acc_n"];
        IMU_MODULE.acc_w_ = imu_node["acc_w"];
        IMU_MODULE.gyr_n_ = imu_node["gyr_n"];
        IMU_MODULE.gyr_w_ = imu_node["gyr_w"];

        G.z() = imu_node["g_norm"];
    }
    else
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    if (ESTIMATE_EXTRINSIC)
    {
        ROS_WARN(" Optimize extrinsic param around initial guess!");
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        ROS_WARN(" fix extrinsic param ");
    }

    cv::FileNode cam_module_node = fsSettings["cam_module"];
    int num_cam_module = cam_module_node["num"];
    printf("camera module number %d\n", num_cam_module);
    CAM_MODULES.resize(num_cam_module);

    cv::FileNode cam_modules_node = cam_module_node["modules"];
    int cur_cam_module = 0;
    for (cv::FileNodeIterator it = cam_modules_node.begin(); it != cam_modules_node.end(); it++, cur_cam_module++)
    {

        if (cur_cam_module >= num_cam_module)
            break;

        CAM_MODULES[cur_cam_module].module_id_ = (*it)["cam_id"];
        int use_depth = (*it)["depth"];
        CAM_MODULES[cur_cam_module].depth_ = use_depth;
        int use_stereo = (*it)["stereo"];
        CAM_MODULES[cur_cam_module].stereo_ = use_stereo;

        CAM_MODULES[cur_cam_module].set_size();

        CAM_MODULES[cur_cam_module].img_width_ = (*it)["image_width"];
        CAM_MODULES[cur_cam_module].img_height_ = (*it)["image_height"];
        CAM_MODULES[cur_cam_module].num_downsamples_ = std::max(0, (int)(*it)["num_downsamples"]);

        CAM_MODULES[cur_cam_module].td_ = (*it)["td"];

        if (ESTIMATE_TD)
        {
            ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << CAM_MODULES[cur_cam_module].td_);
        }
        else
        {
            ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << CAM_MODULES[cur_cam_module].td_);
        }

        int rolling_shutter = (*it)["rolling_shutter"];

        if (rolling_shutter)
        {
            CAM_MODULES[cur_cam_module].tr_ = (*it)["rolling_shutter_tr"];
            ROS_INFO_STREAM("Rolling shutter camera, read out time per line: " << CAM_MODULES[cur_cam_module].tr_);
        }
        else
        {
            CAM_MODULES[cur_cam_module].tr_ = 0.0;
            ROS_INFO("Global shutter camera.");
        }

        (*it)["image0_topic"] >> CAM_MODULES[cur_cam_module].img_topic_[0];
        (*it)["cam0_calib"] >> CAM_MODULES[cur_cam_module].calib_file_[0];
        CAM_MODULES[cur_cam_module].calib_file_[0] = configPath + "/" + CAM_MODULES[cur_cam_module].calib_file_[0];

        cv::Mat cv_T;
        (*it)["imu_T_cam0"] >> cv_T; // pt_in_imu = [R, t] * pt_in_cam0
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        CAM_MODULES[cur_cam_module].ric_[0] = T.block<3, 3>(0, 0);
        CAM_MODULES[cur_cam_module].tic_[0] = T.block<3, 1>(0, 3);

        if (use_depth || use_stereo)
        {
            (*it)["image1_topic"] >> CAM_MODULES[cur_cam_module].img_topic_[1];
            if (use_stereo)
            {
                (*it)["cam1_calib"] >> CAM_MODULES[cur_cam_module].calib_file_[1];
                CAM_MODULES[cur_cam_module].calib_file_[1] = configPath + "/" + CAM_MODULES[cur_cam_module].calib_file_[1];

                cv::Mat cv_T1;
                (*it)["imu_T_cam1"] >> cv_T1;
                Eigen::Matrix4d T1;
                cv::cv2eigen(cv_T1, T1);
                CAM_MODULES[cur_cam_module].ric_[1] = T1.block<3, 3>(0, 0);
                CAM_MODULES[cur_cam_module].tic_[1] = T1.block<3, 1>(0, 3);
            }
        }
    }

    ROS_WARN("%d camera modules:", CAM_MODULES.size());
    for (int i = 0; i < CAM_MODULES.size(); i++)
    {
        cout << "---------------------------" << endl;
        cout << "Cam id: " << CAM_MODULES[i].module_id_ << endl;
        cout << "depth: " << CAM_MODULES[i].depth_ << endl;
        cout << "stereo: " << CAM_MODULES[i].stereo_ << endl;
        cout << "img_dim: " << CAM_MODULES[i].img_width_ << "\t" << CAM_MODULES[i].img_height_ << endl;
        cout << "num_downsamples: " << CAM_MODULES[i].num_downsamples_ << endl;
        cout << "img_topic_0: " << CAM_MODULES[i].img_topic_[0] << endl;
        cout << "calib_file_0: " << CAM_MODULES[i].calib_file_[0] << endl;

        cout << "ric0:\n"
             << CAM_MODULES[i].ric_[0].toRotationMatrix() << endl;
        cout << "tic0: " << CAM_MODULES[i].tic_[0].transpose() << endl;

        if (CAM_MODULES[i].depth_ || CAM_MODULES[i].stereo_)
        {
            cout << "img_topic_1: " << CAM_MODULES[i].img_topic_[1] << endl;
            if (CAM_MODULES[i].stereo_)
            {
                cout << "calib_file_1: " << CAM_MODULES[i].calib_file_[1] << endl;
                cout << "ric1:\n"
                     << CAM_MODULES[i].ric_[1].toRotationMatrix() << endl;
                cout << "tic1: " << CAM_MODULES[i].tic_[1].transpose() << endl;
            }
        }
    }

    // DEPTH = fsSettings["depth"];
    // printf("USE_DEPTH: %d\n", DEPTH);

    // STEREO = fsSettings["stereo"];
    // printf("USE_STEREO: %d\n", STEREO);

    // std::string image_topic;
    // fsSettings["image_topic"] >> image_topic;
    // image_topic.erase(std::remove(image_topic.begin(), image_topic.end(),' '), image_topic.end());

    // stringstream img_topic_str(image_topic);
    // ROS_INFO("image_topic:");
    // int cam_module_id = 0;
    // for(string img_topic_per_module; img_topic_str.good();cam_module_id++){

    //     getline( img_topic_str, img_topic_per_module,';');
    //     stringstream img_topic_per_module_str(img_topic_per_module);
    //     cout<<"module: "<<img_topic_per_module<<endl;
    //     cout<<"cam_module_id: "<<cam_module_id<<endl;
    //     IMAGE_TOPICS.emplace_back(vector<string>());

    //     for(string img_topic; img_topic_per_module_str.good();){
    //         cout<<"module to parse: "<<img_topic_per_module_str.str()<<endl;
    //         getline( img_topic_per_module_str, img_topic, ',');
    //         cout<<"image: "<<img_topic<<endl;
    //         IMAGE_TOPICS[cam_module_id].emplace_back( img_topic );
    //     }
    // }
    // for(auto ss : IMAGE_TOPICS){
    //     cout<<"cam module:\n";
    //     for(auto sss : ss)
    //         cout<<sss<<endl;
    // }

    // std::string cam0Calib;
    // fsSettings["cam0_calib"] >> cam0Calib;
    // stringstream cam0_calib_str(cam0Calib);
    // while( cam0_calib_str.good() ){
    //     string cam0_calib_file;
    //     getline( cam0_calib_str, cam0_calib_file, ',' );
    //     std::string cam0Path = configPath + "/" + cam0Calib;
    //     CAM0_NAMES.push_back(cam0Path);
    // }

    // if(NUM_OF_CAM != IMAGE0_TOPICS.size())
    // {
    //     ROS_ERROR("num_of_cam should be the same as num of image0_topic!");
    //     assert(0);
    // }

    fsSettings["debug_level"] >> DEBUG_LEVEL;

    CV_NUM_THREADS = fsSettings["cv_num_threads"];

    WINDOW_SIZE = fsSettings["window_size"];
    printf("WINDOW_SIZE: %d\n", WINDOW_SIZE);

    MIN_TRACK_NUM_PER_MODULE = fsSettings["min_track_num_per_module"];
    printf("MIN_TRACK_NUM_PER_MODULE: %d\n", MIN_TRACK_NUM_PER_MODULE);

    MIN_OPT_INTERVAL = fsSettings["min_opt_interval"];
    printf("MIN_OPT_INTERVAL: %lf\n", MIN_OPT_INTERVAL);

    MIN_FRAME_INTERVAL_FOR_OPT = fsSettings["min_frame_interval_for_opt"];
    printf("MIN_FRAME_INTERVAL_FOR_OPT: %lf\n", MIN_FRAME_INTERVAL_FOR_OPT);

    MIN_FRAME_INTERVAL_PER_MODULE = fsSettings["min_frame_interval_per_module"];
    printf("MIN_FRAME_INTERVAL_PER_MODULE: %lf\n", MIN_FRAME_INTERVAL_PER_MODULE);

    MIN_TRACK_FRAME_FOR_OPT = fsSettings["min_track_frame_for_opt"];
    printf("MIN_TRACK_FRAME_FOR_OPT: %d\n", MIN_TRACK_FRAME_FOR_OPT);

    MAX_CNT = fsSettings["max_cnt"];
    int max_feature_per_module = max(static_cast<int>(ceil(MAX_CNT / static_cast<double>(CAM_MODULES.size()))), MIN_TRACK_NUM_PER_MODULE);
    MAX_CNT = max_feature_per_module * CAM_MODULES.size();
    MAX_TRACK_NUM_PER_MODULE = MAX_CNT - (CAM_MODULES.size() - 1) * MIN_TRACK_NUM_PER_MODULE;
    printf("MAX_CNT: %d\n", MAX_CNT);

    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    EQUALIZE = fsSettings["equalize"];

    CLAHE_CLIP_LIMIT = fsSettings["clahe_clip_limit"];
    CLAHE_GRID_SIZE = fsSettings["clahe_grid_size"];

    OPTFLOW_WIN_SIZE = fsSettings["optflow_win_size"];
    OPTFLOW_PYR_LEVELS = fsSettings["optflow_pyr_levels"];

    DEPTH_MIN = fsSettings["depth_min"];
    DEPTH_MAX = fsSettings["depth_max"];
    printf("DEPTH_MIN: %lf\n", DEPTH_MIN);
    printf("DEPTH_MAX: %lf\n", DEPTH_MAX);

    NEW_FEATURE_RATIO_THRESHOLD = fsSettings["new_feature_ratio_threshold"];
    printf("NEW_FEATURE_RATIO_THRESHOLD: %lf\n", NEW_FEATURE_RATIO_THRESHOLD);

    MAX_IMG_BUF_SIZE = fsSettings["max_image_buf_size"];
    printf("MAX_IMG_BUF_SIZE: %d\n", MAX_IMG_BUF_SIZE);

    GOOD_FEAT_TO_TRACK_QUALITY = fsSettings["good_feat_to_track_quality"];
    printf("GOOD_FEAT_TO_TRACK_QUALITY: %lf\n", GOOD_FEAT_TO_TRACK_QUALITY);

    IMG_FREQ = fsSettings["img_freq"];
    printf("IMG_FREQ: %lf\n", IMG_FREQ);

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    printf("MIN_PARALLAX: %lf pixels\n", MIN_PARALLAX);
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    VIS_CIRCLE_RADIUS = fsSettings["vis_circle_radius"];
    VIS_ARROW_THICKNESS = fsSettings["vis_arrow_thickness"];

    // ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    // if (ESTIMATE_EXTRINSIC == 2)
    // {
    //     ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
    //     RIC.push_back(Eigen::Matrix3d::Identity());
    //     TIC.push_back(Eigen::Vector3d::Zero());
    //     EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    // }
    // else
    // {
    //     if ( ESTIMATE_EXTRINSIC == 1)
    //     {
    //         ROS_WARN(" Optimize extrinsic param around initial guess!");
    //         EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    //     }
    //     if (ESTIMATE_EXTRINSIC == 0)
    //         ROS_WARN(" fix extrinsic param ");

    //     for(int i  = 0; i < NUM_OF_CAM; i++){
    //         cv::Mat cv_T;
    //         stringstream T_name;
    //         T_name << "body_T_cam0_" << i;
    //         fsSettings[T_name.str().c_str()] >> cv_T;
    //         Eigen::Matrix4d T;
    //         cv::cv2eigen(cv_T, T);
    //         RIC.push_back(T.block<3, 3>(0, 0));
    //         TIC.push_back(T.block<3, 1>(0, 3));
    //     }
    // }

    // if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    // {
    //     printf("num_of_cam should be 1 or 2\n");
    //     assert(0);
    // }

    // if(STEREO || DEPTH){
    //     std::string image1_topic;
    //     fsSettings["image1_topic"] >> image1_topic;
    //     stringstream img1_topic_str(image1_topic);
    //     while( img1_topic_str.good() ){
    //         string img1_topic;
    //         getline( img1_topic_str, img1_topic, ',' );
    //         IMAGE1_TOPICS.push_back( img1_topic );
    //     }
    //     for(auto ss : IMAGE1_TOPICS)
    //         cout<<ss<<endl;

    //     if(NUM_OF_CAM != IMAGE1_TOPICS.size())
    //     {
    //         ROS_ERROR("num_of_cam should be the same as num of image1_topic!");
    //         assert(0);
    //     }

    //     if(STEREO){
    //         std::string cam1Calib;
    //         fsSettings["cam1_calib"] >> cam1Calib;
    //         std::string cam1Path = configPath + "/" + cam1Calib;
    //         //printf("%s cam1 path\n", cam1Path.c_str() );
    //         CAM1_NAMES.push_back(cam1Path);

    //         for(int i  = 0; i < NUM_OF_CAM; i++){
    //             cv::Mat cv_T;
    //             stringstream T_name;
    //             T_name << "body_T_cam1_" << i;
    //             fsSettings[T_name.str().c_str()] >> cv_T;
    //             Eigen::Matrix4d T;
    //             cv::cv2eigen(cv_T, T);
    //             RIC1.push_back(T.block<3, 3>(0, 0));
    //             TIC1.push_back(T.block<3, 1>(0, 3));
    //         }
    //     }

    //     if(DEPTH){
    //         DEPTH_MIN = fsSettings["depth_min"];
    //         DEPTH_MAX = fsSettings["depth_max"];
    //         printf("DEPTH_MIN: %lf\n", DEPTH_MIN);
    //         printf("DEPTH_MAX: %lf\n", DEPTH_MAX);
    //     }
    // }

    // cv::Mat cv_center_T_imu;
    // fsSettings["center_T_imu"] >> cv_center_T_imu;
    // Eigen::Matrix4d T_temp;
    // cv::cv2eigen(cv_center_T_imu, T_temp);
    // center_R_imu = T_temp.block<3, 3>(0, 0);
    // center_T_imu = T_temp.block<3, 1>(0, 3);

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    // ROW = fsSettings["image_height"];
    // COL = fsSettings["image_width"];
    // ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    // if(!USE_IMU)
    // {
    //     ESTIMATE_EXTRINSIC = 0;
    //     ESTIMATE_TD = 0;
    //     printf("no imu, fix extrinsic param; no time offset calibration\n");
    // }

    fsSettings.release();
}

} // namespace vins_multi