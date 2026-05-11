#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <ros/ros.h>

namespace gloc
{

extern int GLOC_ENABLED;
extern std::string GLOC_COLMAP_FOLDER;
extern std::string GLOC_COLMAP_IMG_FOLDER;
extern std::string GLOC_DBOW3_DATABASE;
extern std::string GLOC_DBOW3_VOCAB;

void readParameters(std::string config_file);

}; // namespace gloc