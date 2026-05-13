#include "parameters.h"

namespace gloc
{

int GLOC_ENABLED;
std::string GLOC_COLMAP_FOLDER;
std::string GLOC_COLMAP_IMG_FOLDER;
std::string GLOC_DBOW3_DATABASE;
std::string GLOC_DBOW3_VOCAB;

// DBoW3
int GLOC_DBOW3_MAX_RESULTS = 5;
double GLOC_DBOW3_MIN_SCORE = 0.01;

// Consensus voting
double GLOC_VOTE_EPS_M = 1.0;
int GLOC_VOTE_MIN_VOTES = -1; // -1 → majority rule at runtime

// ORB
int GLOC_ORB_NFEATURES = 500;
float GLOC_ORB_SCALE_FACTOR = 1.2f;
int GLOC_ORB_NLEVELS = 8;
int GLOC_ORB_EDGE_THRESHOLD = 31;
int GLOC_ORB_FIRST_LEVEL = 0;
int GLOC_ORB_WTA_K = 2;
int GLOC_ORB_SCORE_TYPE = 0; // 0 = HARRIS
int GLOC_ORB_PATCH_SIZE = 31;
int GLOC_ORB_FAST_THRESHOLD = 20;

// Descriptor
int GLOC_USE_BEBLID = 0;
float GLOC_BEBLID_SCALE_FACTOR = 1.0f;
int GLOC_BEBLID_N_BITS = 256;

// Matching
int GLOC_USE_GMS = 1;
float GLOC_GMS_THRESHOLD = 6.0f;
int GLOC_GMS_WITH_ROTATION = 0;
int GLOC_GMS_WITH_SCALE = 0;

// Matching
float GLOC_MATCH_LOWE_RATIO = 0.8f;
int GLOC_MATCH_MAX_DIST = 64;
float GLOC_MATCH_GEOM_REPROJ_TH = 3.0f;
float GLOC_MATCH_GEOM_CONFIDENCE = 0.99f;
float GLOC_MATCH_GEOM_SAMPSON_SQ = 4.0f;
int GLOC_MATCH_MIN_INLIERS = 10;

// Helper: read a value only when the node is non-empty/non-null.
template <typename T>
static void read_if(const cv::FileStorage &fs,
                    const std::string &key, T &out)
{
    const cv::FileNode n = fs[key];
    if (!n.empty())
        n >> out;
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

    gloc::GLOC_ENABLED = fsSettings["gloc_enabled"];

    fsSettings["gloc_colmap_folder"] >> gloc::GLOC_COLMAP_FOLDER;
    fsSettings["gloc_colmap_img_folder"] >> gloc::GLOC_COLMAP_IMG_FOLDER;
    fsSettings["gloc_dbow3_database"] >> gloc::GLOC_DBOW3_DATABASE;
    fsSettings["gloc_dbow3_vocab"] >> gloc::GLOC_DBOW3_VOCAB;

    read_if(fsSettings, "gloc_dbow3_max_results", gloc::GLOC_DBOW3_MAX_RESULTS);
    read_if(fsSettings, "gloc_dbow3_min_score", gloc::GLOC_DBOW3_MIN_SCORE);

    read_if(fsSettings, "gloc_vote_eps_m", gloc::GLOC_VOTE_EPS_M);
    read_if(fsSettings, "gloc_vote_min_votes", gloc::GLOC_VOTE_MIN_VOTES);

    read_if(fsSettings, "gloc_orb_nfeatures", gloc::GLOC_ORB_NFEATURES);
    read_if(fsSettings, "gloc_orb_scale_factor", gloc::GLOC_ORB_SCALE_FACTOR);
    read_if(fsSettings, "gloc_orb_nlevels", gloc::GLOC_ORB_NLEVELS);
    read_if(fsSettings, "gloc_orb_edge_threshold", gloc::GLOC_ORB_EDGE_THRESHOLD);
    read_if(fsSettings, "gloc_orb_first_level", gloc::GLOC_ORB_FIRST_LEVEL);
    read_if(fsSettings, "gloc_orb_wta_k", gloc::GLOC_ORB_WTA_K);
    read_if(fsSettings, "gloc_orb_score_type", gloc::GLOC_ORB_SCORE_TYPE);
    read_if(fsSettings, "gloc_orb_patch_size", gloc::GLOC_ORB_PATCH_SIZE);
    read_if(fsSettings, "gloc_orb_fast_threshold", gloc::GLOC_ORB_FAST_THRESHOLD);

    read_if(fsSettings, "gloc_use_beblid", gloc::GLOC_USE_BEBLID);
    read_if(fsSettings, "gloc_beblid_scale_factor", gloc::GLOC_BEBLID_SCALE_FACTOR);
    read_if(fsSettings, "gloc_beblid_n_bits", gloc::GLOC_BEBLID_N_BITS);

    read_if(fsSettings, "gloc_use_gms", gloc::GLOC_USE_GMS);
    read_if(fsSettings, "gloc_gms_threshold", gloc::GLOC_GMS_THRESHOLD);
    read_if(fsSettings, "gloc_gms_with_rotation", gloc::GLOC_GMS_WITH_ROTATION);
    read_if(fsSettings, "gloc_gms_with_scale", gloc::GLOC_GMS_WITH_SCALE);

    read_if(fsSettings, "gloc_match_lowe_ratio", gloc::GLOC_MATCH_LOWE_RATIO);
    read_if(fsSettings, "gloc_match_max_dist", gloc::GLOC_MATCH_MAX_DIST);
    read_if(fsSettings, "gloc_match_geom_reproj_th", gloc::GLOC_MATCH_GEOM_REPROJ_TH);
    read_if(fsSettings, "gloc_match_geom_confidence", gloc::GLOC_MATCH_GEOM_CONFIDENCE);
    read_if(fsSettings, "gloc_match_geom_sampson_sq", gloc::GLOC_MATCH_GEOM_SAMPSON_SQ);
    read_if(fsSettings, "gloc_match_min_inliers", gloc::GLOC_MATCH_MIN_INLIERS);

    printf("GLOC_ENABLED            : %d\n", gloc::GLOC_ENABLED);
    printf("GLOC_COLMAP_FOLDER      : %s\n", gloc::GLOC_COLMAP_FOLDER.c_str());
    printf("GLOC_COLMAP_IMG_FOLDER  : %s\n", gloc::GLOC_COLMAP_IMG_FOLDER.c_str());
    printf("GLOC_DBOW3_DATABASE     : %s\n", gloc::GLOC_DBOW3_DATABASE.c_str());
    printf("GLOC_DBOW3_VOCAB        : %s\n", gloc::GLOC_DBOW3_VOCAB.c_str());
    printf("GLOC_DBOW3_MAX_RESULTS  : %d\n", gloc::GLOC_DBOW3_MAX_RESULTS);
    printf("GLOC_DBOW3_MIN_SCORE    : %.4f\n", gloc::GLOC_DBOW3_MIN_SCORE);
    printf("GLOC_VOTE_EPS_M         : %.2f\n", gloc::GLOC_VOTE_EPS_M);
    printf("GLOC_VOTE_MIN_VOTES     : %d\n", gloc::GLOC_VOTE_MIN_VOTES);
    printf("GLOC_USE_BEBLID         : %d\n", gloc::GLOC_USE_BEBLID);
    printf("GLOC_BEBLID_SCALE_FACTOR: %.2f\n", gloc::GLOC_BEBLID_SCALE_FACTOR);
    printf("GLOC_BEBLID_N_BITS      : %d\n", gloc::GLOC_BEBLID_N_BITS);
    printf("GLOC_USE_GMS            : %d\n", gloc::GLOC_USE_GMS);
    printf("GLOC_GMS_THRESHOLD      : %.2f\n", gloc::GLOC_GMS_THRESHOLD);
    printf("GLOC_GMS_WITH_ROTATION  : %d\n", gloc::GLOC_GMS_WITH_ROTATION);
    printf("GLOC_GMS_WITH_SCALE     : %d\n", gloc::GLOC_GMS_WITH_SCALE);
    printf("GLOC_ORB_NFEATURES      : %d\n", gloc::GLOC_ORB_NFEATURES);
    printf("GLOC_MATCH_MIN_INLIERS  : %d\n", gloc::GLOC_MATCH_MIN_INLIERS);
}

}; // namespace gloc