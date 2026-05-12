/*******************************************************
 * Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "ros/node_handle.h"
#include "ros/subscriber.h"
#include "utility/tic_toc.h"
#include "utility/visualization.h"
#include <cv_bridge/cv_bridge.h>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <ros/ros.h>
#include <stdio.h>

#include <boost/thread.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CompressedImage.h>
#include <vector>

#include "estimator/parameters.h"
#include "gloc/gloc.h"
#include "gloc/parameters.h"

using namespace std;

namespace vins_multi
{

class VinsNodeBaseClass
{
  public:
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicyImageExact;
    typedef unique_ptr<message_filters::Synchronizer<SyncPolicyImageExact>> SynchronizerImageExact;

    struct uni_image_sub_ptr
    {
        message_filters::Subscriber<sensor_msgs::Image> *img0_sync_sub_ptr_ = nullptr;
        message_filters::Subscriber<sensor_msgs::Image> *img1_sync_sub_ptr_ = nullptr;

        SynchronizerImageExact sync_image_exact_;

        // Mono: single-image subscriber (value, not pointer)
        ros::Subscriber img0_sub_;
    };

    class camera_module_info_with_sub
    {
      public:
        camera_module_info_with_sub(camera_module_info cam_module,
                                    Estimator *est_ptr,
                                    gloc::Gloc *gloc_ptr)
            : module_info_(cam_module), estimator_ptr_(est_ptr), gloc_ptr_(gloc_ptr)
        {
        }

        uni_image_sub_ptr img_sub_;

        camera_module_info module_info_;

        // VINS-side unique_id: the index into CAM_MODULES.
        unsigned int unique_id_;

        // Gloc-side unique_id: the index into GLOC_CAM_MODULES, OR -1 if this
        // physical camera is not also flagged use_for_gloc. Resolved once at
        // setup by matching module_id_ against GLOC_CAM_MODULES. Used by the
        // callback to push the cam0 image into the gloc ring buffer when this
        // camera is dual-use (use_for_vins:1 + use_for_gloc:1).
        int gloc_unique_id_ = -1;

        double last_img_t_ = -1.0;

        Estimator *estimator_ptr_;
        gloc::Gloc *gloc_ptr_;

        void imgs_callback(const sensor_msgs::ImageConstPtr &img0_msg, const sensor_msgs::ImageConstPtr &img1_msg);

        void img_callback(const sensor_msgs::ImageConstPtr &img0_msg);

        void comp_imgs_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg, const sensor_msgs::CompressedImageConstPtr &img2_msg);
    };

    // ─────────────────────────────────────────────────────────────────────────
    // gloc_only_camera_module_info_with_sub
    //
    // Subscribes for a camera flagged use_for_gloc:1 but NOT use_for_vins:1.
    // Its callbacks decode the incoming image identically to the VINS path
    // (grayscale conversion, downsampling) but route only into the gloc ring
    // buffer — no estimator dispatch.
    //
    // For stereo entries, only cam0 (left) is consumed; cam1 is not even
    // subscribed (gloc uses the left image).
    // ─────────────────────────────────────────────────────────────────────────
    class gloc_only_camera_module_info_with_sub
    {
      public:
        gloc_only_camera_module_info_with_sub(camera_module_info cam_module,
                                              gloc::Gloc *gloc_ptr)
            : module_info_(cam_module), gloc_ptr_(gloc_ptr)
        {
        }

        // Only cam0 is subscribed — gloc consumes the left image only, so
        // there is no stereo synchronizer here.
        ros::Subscriber img0_sub_;

        camera_module_info module_info_;

        // Gloc-side unique_id: the index into GLOC_CAM_MODULES.
        unsigned int gloc_unique_id_;

        double last_img_t_ = -1.0;

        gloc::Gloc *gloc_ptr_;

        // Same rate-gating, color-conversion, and downsampling as the VINS
        // callback; result is pushed to gloc only.
        void img_callback(const sensor_msgs::ImageConstPtr &img0_msg);
    };

    class imu_info_with_sub
    {
      public:
        imu_info_with_sub(imu_info imu_module, Estimator *est_ptr) : module_info_(imu_module), estimator_ptr_(est_ptr)
        {
        }

        ros::Subscriber imu_sub_;
        imu_info module_info_;
        Estimator *estimator_ptr_;

        void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg);
    };

    void init_node(ros::NodeHandle &n, const std::string &config_file);

  private:
    vector<camera_module_info_with_sub> camera_modules_;
    // Subscribers for cameras flagged use_for_gloc:1 && !use_for_vins:1.
    // Cameras in BOTH lists are handled by camera_modules_ above (which also
    // pushes into gloc when gloc_unique_id_ >= 0), so this vector contains
    // only the "gloc-only" subset.
    vector<gloc_only_camera_module_info_with_sub> gloc_only_camera_modules_;
    vector<imu_info_with_sub> imu_modules_;

    Estimator estimator_;
    gloc::Gloc gloc_;

    ros::Subscriber sub_restart_;

  protected:
    void registerSub(ros::NodeHandle &n);

    void set_modules();

    void restart_callback(const std_msgs::BoolConstPtr &restart_msg);

    virtual void Init(ros::NodeHandle &n, const std::string &config_file);
};

} // namespace vins_multi