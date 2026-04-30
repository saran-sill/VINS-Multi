/*******************************************************
 * Copyright (C) 2025, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include "rosNode.hpp"
#include "estimator/parameters.h"

int main(int argc, char **argv)
{
    std::string cli_config_file;
    if (argc >= 3)
    {
        cli_config_file = argv[2];
    }
    else
    {
        printf("usage : %s [device_name] [config.yaml]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string device_name = std::string(argv[1]);

    ros::init(argc, argv, device_name);
    ros::NodeHandle n("~");

    vins_multi::VinsNodeBaseClass vins_node;

    vins_node.init_node(n, cli_config_file);

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (vins_multi::DEBUG_LEVEL == "debug")
    {
        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    }

     cv::setNumThreads(vins_multi::CV_NUM_THREADS);

#ifdef _OPENMP
    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(vins_multi::CV_NUM_THREADS); // Use 4 threads for all consecutive parallel regions
#endif

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ros::spin();

    return 0;
}
