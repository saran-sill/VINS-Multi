#include "parameters.h"

namespace gloc
{

int GLOC_ENABLED;
std::string GLOC_COLMAP_FOLDER;
std::string GLOC_COLMAP_IMG_FOLDER;
std::string GLOC_DBOW3_DATABASE;
std::string GLOC_DBOW3_VOCAB;

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

    gloc::GLOC_ENABLED = fsSettings["gloc_enabled"];

    fsSettings["gloc_colmap_folder"]     >> gloc::GLOC_COLMAP_FOLDER;
    fsSettings["gloc_colmap_img_folder"] >> gloc::GLOC_COLMAP_IMG_FOLDER;
    fsSettings["gloc_dbow3_database"]    >> gloc::GLOC_DBOW3_DATABASE;
    fsSettings["gloc_dbow3_vocab"]       >> gloc::GLOC_DBOW3_VOCAB;

    printf("GLOC_ENABLED          : %d\n",   gloc::GLOC_ENABLED);
    printf("GLOC_COLMAP_FOLDER    : %s\n",   gloc::GLOC_COLMAP_FOLDER.c_str());
    printf("GLOC_COLMAP_IMG_FOLDER: %s\n",   gloc::GLOC_COLMAP_IMG_FOLDER.c_str());
    printf("GLOC_DBOW3_DATABASE   : %s\n",   gloc::GLOC_DBOW3_DATABASE.c_str());
    printf("GLOC_DBOW3_VOCAB      : %s\n",   gloc::GLOC_DBOW3_VOCAB.c_str());
}

}; // namespace gloc