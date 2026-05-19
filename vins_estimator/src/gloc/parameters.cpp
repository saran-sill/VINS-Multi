#include "parameters.h"

#include <filesystem>

namespace gloc
{

bool GLOC_ENABLED = false;
std::string GLOC_COLMAP_SPARSE_FOLDER;
std::string GLOC_COLMAP_IMG_FOLDER;
std::string GLOC_DBOW3_DATABASE;
std::string GLOC_DBOW3_VOCAB;
std::string GLOC_WORLD_TMAT_FILE;

// DBoW3
int GLOC_DBOW3_MAX_RESULTS = 5;
double GLOC_DBOW3_MIN_SCORE = 0.01;

// Consensus voting
double GLOC_VOTE_EPS_M = 1.0;
int GLOC_VOTE_MIN_VOTES = -1;
double GLOC_VOTE_MAX_DIST_M = -1.0;
int GLOC_VOTE_MAX_MATCHES = 1;
double GLOC_STARTUP_DELAY_S = 0.0;

// ORB
int GLOC_ORB_NFEATURES = 500;
float GLOC_ORB_SCALE_FACTOR = 1.2f;
int GLOC_ORB_NLEVELS = 8;
int GLOC_ORB_EDGE_THRESHOLD = 31;
int GLOC_ORB_FIRST_LEVEL = 0;
int GLOC_ORB_WTA_K = 2;
int GLOC_ORB_SCORE_TYPE = 0;
int GLOC_ORB_PATCH_SIZE = 31;
int GLOC_ORB_FAST_THRESHOLD = 20;

// Descriptor
bool GLOC_USE_BEBLID = false;
float GLOC_BEBLID_SCALE_FACTOR = 1.0f;
int GLOC_BEBLID_N_BITS = 256;

// Matching
bool GLOC_USE_GMS = true;
float GLOC_GMS_THRESHOLD = 6.0f;
bool GLOC_GMS_WITH_ROTATION = false;
bool GLOC_GMS_WITH_SCALE = false;

float GLOC_MATCH_LOWE_RATIO = 0.8f;
int GLOC_MATCH_MAX_DIST = 64;
float GLOC_MATCH_GEOM_REPROJ_TH = 3.0f;
float GLOC_MATCH_GEOM_CONFIDENCE = 0.99f;
float GLOC_MATCH_GEOM_SAMPSON_SQ = 4.0f;
int GLOC_MATCH_MIN_INLIERS = 10;
float GLOC_SUBSAMPLE_MIN_DIST_PX = 0.0f;

// Stage 2 — Ceres optimization
int GLOC_MIN_PAIRS = 3;
int GLOC_MIN_PAIRS_FIRST_SNAP = 8;
int GLOC_REL_POSE_K = 1;
double GLOC_W_EPIPOLAR = 1.0;
double GLOC_W_REPROJ = 1.0;
double GLOC_W_REL_POSE = 1.0;
double GLOC_W_WORLD_PRIOR = 10.0;
double GLOC_HUBER_DELTA = 5.0;
double GLOC_INLIER_THRESH_PX = 10.0;
int GLOC_MAX_ITERS = 150;
int GLOC_INIT_ITERS = 20;
double GLOC_MIN_INLIER_RATIO = 0.3;
double GLOC_MAX_DEPTH_M = 200.0;
bool GLOC_USE_4DOF = false;
bool GLOC_FIX_REL_POSES = false;
std::string GLOC_DEBUG_FOLDER;

// Preprocessing
bool GLOC_PREPROCESS_WHITE_BALANCE = false;
bool GLOC_PREPROCESS_DENOISE = false;
int GLOC_PREPROCESS_DENOISE_D = 5;
double GLOC_PREPROCESS_DENOISE_SIGMA_COLOR = 40.0;
double GLOC_PREPROCESS_DENOISE_SIGMA_SPACE = 40.0;
bool GLOC_PREPROCESS_DENOISE_COLOR = false;
double GLOC_PREPROCESS_DENOISE_H_LUMINANCE = 2.5;
double GLOC_PREPROCESS_DENOISE_H_COLOR = 10.0;
bool GLOC_PREPROCESS_GAMMA = false;
double GLOC_PREPROCESS_GAMMA_VALUE = 2.2;
bool GLOC_PREPROCESS_TONEMAP = false;
double GLOC_PREPROCESS_TONEMAP_GAMMA = 0.78;
double GLOC_PREPROCESS_TONEMAP_HIGHLIGHT = 0.65;
double GLOC_PREPROCESS_TONEMAP_SHADOW_LIFT = 0.08;
bool GLOC_PREPROCESS_CLAHE = true;
double GLOC_PREPROCESS_CLAHE_CLIP_LIMIT = 5.0;
int GLOC_PREPROCESS_CLAHE_GRID_SIZE = 5;
bool GLOC_PREPROCESS_CLARITY = false;
double GLOC_PREPROCESS_CLARITY_AMOUNT = 0.40;
double GLOC_PREPROCESS_CLARITY_SIGMA = 15.0;
bool GLOC_PREPROCESS_SHARPEN = false;
double GLOC_PREPROCESS_SHARPEN_SIGMA = 1.0;
double GLOC_PREPROCESS_SHARPEN_AMOUNT = 1.5;

// Helper: read a value only when the node is non-empty/non-null.
template <typename T>
static void read_if(const cv::FileStorage &fs,
                    const std::string &key, T &out)
{
    const cv::FileNode n = fs[key];
    if (!n.empty())
        n >> out;
}

// Bool specialization: read as int (0/1) then convert.
// Supports both YAML boolean (true/false) and integer (0/1).
template <>
void read_if<bool>(const cv::FileStorage &fs,
                   const std::string &key, bool &out)
{
    const cv::FileNode n = fs[key];
    if (n.empty())
        return;
    if (n.isInt())
        out = static_cast<int>(n) != 0;
    else if (n.isString())
    {
        std::string s = static_cast<std::string>(n);
        // Trim at first whitespace or '#'
        const auto cut = s.find_first_of(" \t#");
        if (cut != std::string::npos)
            s = s.substr(0, cut);
        out = (s == "true" || s == "1" || s == "yes");
    }
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

    read_if(fsSettings, "gloc_enabled", gloc::GLOC_ENABLED);

    fsSettings["gloc_colmap_sparse_folder"] >> gloc::GLOC_COLMAP_SPARSE_FOLDER;
    fsSettings["gloc_colmap_img_folder"] >> gloc::GLOC_COLMAP_IMG_FOLDER;
    fsSettings["gloc_dbow3_database"] >> gloc::GLOC_DBOW3_DATABASE;
    fsSettings["gloc_dbow3_vocab"] >> gloc::GLOC_DBOW3_VOCAB;
    read_if(fsSettings, "gloc_world_tmat_file", gloc::GLOC_WORLD_TMAT_FILE);

    read_if(fsSettings, "gloc_dbow3_max_results", gloc::GLOC_DBOW3_MAX_RESULTS);
    read_if(fsSettings, "gloc_dbow3_min_score", gloc::GLOC_DBOW3_MIN_SCORE);
    read_if(fsSettings, "gloc_vote_eps_m", gloc::GLOC_VOTE_EPS_M);
    read_if(fsSettings, "gloc_vote_min_votes", gloc::GLOC_VOTE_MIN_VOTES);
    read_if(fsSettings, "gloc_vote_max_dist_m", gloc::GLOC_VOTE_MAX_DIST_M);
    read_if(fsSettings, "gloc_vote_max_matches", gloc::GLOC_VOTE_MAX_MATCHES);
    read_if(fsSettings, "gloc_startup_delay_s", gloc::GLOC_STARTUP_DELAY_S);
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
    read_if(fsSettings, "gloc_subsample_min_dist_px", gloc::GLOC_SUBSAMPLE_MIN_DIST_PX);
    read_if(fsSettings, "gloc_min_pairs", gloc::GLOC_MIN_PAIRS);
    read_if(fsSettings, "gloc_min_pairs_first_snap", gloc::GLOC_MIN_PAIRS_FIRST_SNAP);
    read_if(fsSettings, "gloc_rel_pose_k", gloc::GLOC_REL_POSE_K);
    read_if(fsSettings, "gloc_w_epipolar", gloc::GLOC_W_EPIPOLAR);
    read_if(fsSettings, "gloc_w_reproj", gloc::GLOC_W_REPROJ);
    read_if(fsSettings, "gloc_w_rel_pose", gloc::GLOC_W_REL_POSE);
    read_if(fsSettings, "gloc_w_world_prior", gloc::GLOC_W_WORLD_PRIOR);
    read_if(fsSettings, "gloc_huber_delta", gloc::GLOC_HUBER_DELTA);
    read_if(fsSettings, "gloc_inlier_thresh_px", gloc::GLOC_INLIER_THRESH_PX);
    read_if(fsSettings, "gloc_max_iters", gloc::GLOC_MAX_ITERS);
    read_if(fsSettings, "gloc_init_iters", gloc::GLOC_INIT_ITERS);
    read_if(fsSettings, "gloc_min_inlier_ratio", gloc::GLOC_MIN_INLIER_RATIO);
    read_if(fsSettings, "gloc_max_depth_m", gloc::GLOC_MAX_DEPTH_M);
    read_if(fsSettings, "gloc_use_4dof", gloc::GLOC_USE_4DOF);
    read_if(fsSettings, "gloc_fix_rel_poses", gloc::GLOC_FIX_REL_POSES);
    read_if(fsSettings, "gloc_debug_folder", gloc::GLOC_DEBUG_FOLDER);

    // Preprocessing
    read_if(fsSettings, "gloc_preprocess_white_balance", gloc::GLOC_PREPROCESS_WHITE_BALANCE);
    read_if(fsSettings, "gloc_preprocess_denoise", gloc::GLOC_PREPROCESS_DENOISE);
    read_if(fsSettings, "gloc_preprocess_denoise_d", gloc::GLOC_PREPROCESS_DENOISE_D);
    read_if(fsSettings, "gloc_preprocess_denoise_sigma_color", gloc::GLOC_PREPROCESS_DENOISE_SIGMA_COLOR);
    read_if(fsSettings, "gloc_preprocess_denoise_sigma_space", gloc::GLOC_PREPROCESS_DENOISE_SIGMA_SPACE);
    read_if(fsSettings, "gloc_preprocess_denoise_color", gloc::GLOC_PREPROCESS_DENOISE_COLOR);
    read_if(fsSettings, "gloc_preprocess_denoise_h_luminance", gloc::GLOC_PREPROCESS_DENOISE_H_LUMINANCE);
    read_if(fsSettings, "gloc_preprocess_denoise_h_color", gloc::GLOC_PREPROCESS_DENOISE_H_COLOR);
    read_if(fsSettings, "gloc_preprocess_gamma", gloc::GLOC_PREPROCESS_GAMMA);
    read_if(fsSettings, "gloc_preprocess_gamma_value", gloc::GLOC_PREPROCESS_GAMMA_VALUE);
    read_if(fsSettings, "gloc_preprocess_tonemap", gloc::GLOC_PREPROCESS_TONEMAP);
    read_if(fsSettings, "gloc_preprocess_tonemap_gamma", gloc::GLOC_PREPROCESS_TONEMAP_GAMMA);
    read_if(fsSettings, "gloc_preprocess_tonemap_highlight", gloc::GLOC_PREPROCESS_TONEMAP_HIGHLIGHT);
    read_if(fsSettings, "gloc_preprocess_tonemap_shadow_lift", gloc::GLOC_PREPROCESS_TONEMAP_SHADOW_LIFT);
    read_if(fsSettings, "gloc_preprocess_clahe", gloc::GLOC_PREPROCESS_CLAHE);
    read_if(fsSettings, "gloc_preprocess_clahe_clip_limit", gloc::GLOC_PREPROCESS_CLAHE_CLIP_LIMIT);
    read_if(fsSettings, "gloc_preprocess_clahe_grid_size", gloc::GLOC_PREPROCESS_CLAHE_GRID_SIZE);
    read_if(fsSettings, "gloc_preprocess_clarity", gloc::GLOC_PREPROCESS_CLARITY);
    read_if(fsSettings, "gloc_preprocess_clarity_amount", gloc::GLOC_PREPROCESS_CLARITY_AMOUNT);
    read_if(fsSettings, "gloc_preprocess_clarity_sigma", gloc::GLOC_PREPROCESS_CLARITY_SIGMA);
    read_if(fsSettings, "gloc_preprocess_sharpen", gloc::GLOC_PREPROCESS_SHARPEN);
    read_if(fsSettings, "gloc_preprocess_sharpen_sigma", gloc::GLOC_PREPROCESS_SHARPEN_SIGMA);
    read_if(fsSettings, "gloc_preprocess_sharpen_amount", gloc::GLOC_PREPROCESS_SHARPEN_AMOUNT);

    // Validate debug folder — clear if it doesn't exist on disk so the
    // empty-string guard in saveDebugImages disables saving automatically.
    if (!gloc::GLOC_DEBUG_FOLDER.empty() &&
        !std::filesystem::exists(gloc::GLOC_DEBUG_FOLDER))
    {
        printf("GLOC_DEBUG_FOLDER '%s' does not exist — debug images disabled.\n",
               gloc::GLOC_DEBUG_FOLDER.c_str());
        gloc::GLOC_DEBUG_FOLDER.clear();
    }
    else if (!gloc::GLOC_DEBUG_FOLDER.empty() &&
             gloc::GLOC_DEBUG_FOLDER.back() != '/')
    {
        gloc::GLOC_DEBUG_FOLDER += '/';
    }

    printf("GLOC_ENABLED              : %d\n", gloc::GLOC_ENABLED);
    printf("GLOC_COLMAP_SPARSE_FOLDER : %s\n", gloc::GLOC_COLMAP_SPARSE_FOLDER.c_str());
    printf("GLOC_COLMAP_IMG_FOLDER    : %s\n", gloc::GLOC_COLMAP_IMG_FOLDER.c_str());
    printf("GLOC_DBOW3_DATABASE       : %s\n", gloc::GLOC_DBOW3_DATABASE.c_str());
    printf("GLOC_DBOW3_VOCAB          : %s\n", gloc::GLOC_DBOW3_VOCAB.c_str());
    printf("GLOC_WORLD_TMAT_FILE      : %s\n", gloc::GLOC_WORLD_TMAT_FILE.c_str());
    printf("GLOC_DBOW3_MAX_RESULTS    : %d\n", gloc::GLOC_DBOW3_MAX_RESULTS);
    printf("GLOC_DBOW3_MIN_SCORE      : %.4f\n", gloc::GLOC_DBOW3_MIN_SCORE);
    printf("GLOC_VOTE_EPS_M           : %.2f\n", gloc::GLOC_VOTE_EPS_M);
    printf("GLOC_VOTE_MIN_VOTES       : %d\n", gloc::GLOC_VOTE_MIN_VOTES);
    printf("GLOC_VOTE_MAX_DIST_M      : %.2f\n", gloc::GLOC_VOTE_MAX_DIST_M);
    printf("GLOC_VOTE_MAX_MATCHES     : %d\n", gloc::GLOC_VOTE_MAX_MATCHES);
    printf("GLOC_STARTUP_DELAY_S      : %.1f\n", gloc::GLOC_STARTUP_DELAY_S);
    printf("GLOC_USE_BEBLID           : %d\n", gloc::GLOC_USE_BEBLID);
    printf("GLOC_BEBLID_SCALE_FACTOR  : %.2f\n", gloc::GLOC_BEBLID_SCALE_FACTOR);
    printf("GLOC_BEBLID_N_BITS        : %d\n", gloc::GLOC_BEBLID_N_BITS);
    printf("GLOC_USE_GMS              : %d\n", gloc::GLOC_USE_GMS);
    printf("GLOC_GMS_THRESHOLD        : %.2f\n", gloc::GLOC_GMS_THRESHOLD);
    printf("GLOC_GMS_WITH_ROTATION    : %d\n", gloc::GLOC_GMS_WITH_ROTATION);
    printf("GLOC_GMS_WITH_SCALE       : %d\n", gloc::GLOC_GMS_WITH_SCALE);
    printf("GLOC_ORB_NFEATURES        : %d\n", gloc::GLOC_ORB_NFEATURES);
    printf("GLOC_MATCH_MIN_INLIERS    : %d\n", gloc::GLOC_MATCH_MIN_INLIERS);
    printf("GLOC_SUBSAMPLE_MIN_DIST   : %.1f\n", gloc::GLOC_SUBSAMPLE_MIN_DIST_PX);
    printf("GLOC_MIN_PAIRS            : %d\n", gloc::GLOC_MIN_PAIRS);
    printf("GLOC_MIN_PAIRS_FIRST_SNAP : %d\n", gloc::GLOC_MIN_PAIRS_FIRST_SNAP);
    printf("GLOC_REL_POSE_K           : %d\n", gloc::GLOC_REL_POSE_K);
    printf("GLOC_W_EPIPOLAR           : %.2f\n", gloc::GLOC_W_EPIPOLAR);
    printf("GLOC_W_REPROJ             : %.2f\n", gloc::GLOC_W_REPROJ);
    printf("GLOC_W_REL_POSE           : %.2f\n", gloc::GLOC_W_REL_POSE);
    printf("GLOC_W_WORLD_PRIOR        : %.2f\n", gloc::GLOC_W_WORLD_PRIOR);
    printf("GLOC_HUBER_DELTA          : %.2f\n", gloc::GLOC_HUBER_DELTA);
    printf("GLOC_INLIER_THRESH_PX     : %.2f\n", gloc::GLOC_INLIER_THRESH_PX);
    printf("GLOC_MAX_ITERS            : %d\n", gloc::GLOC_MAX_ITERS);
    printf("GLOC_INIT_ITERS           : %d\n", gloc::GLOC_INIT_ITERS);
    printf("GLOC_MIN_INLIER_RATIO     : %.2f\n", gloc::GLOC_MIN_INLIER_RATIO);
    printf("GLOC_MAX_DEPTH_M          : %.1f\n", gloc::GLOC_MAX_DEPTH_M);
    printf("GLOC_USE_4DOF             : %d\n", gloc::GLOC_USE_4DOF);
    printf("GLOC_FIX_REL_POSES        : %d\n", gloc::GLOC_FIX_REL_POSES);
    printf("GLOC_DEBUG_FOLDER         : %s\n", gloc::GLOC_DEBUG_FOLDER.c_str());
    printf("GLOC_PREPROCESS_WHITE_BAL : %d\n", gloc::GLOC_PREPROCESS_WHITE_BALANCE);
    printf("GLOC_PREPROCESS_DENOISE   : %d\n", gloc::GLOC_PREPROCESS_DENOISE);
    printf("GLOC_PREPROCESS_TONEMAP   : %d\n", gloc::GLOC_PREPROCESS_TONEMAP);
    printf("GLOC_PREPROCESS_CLAHE     : %d (clip=%.1f grid=%d)\n",
           gloc::GLOC_PREPROCESS_CLAHE,
           gloc::GLOC_PREPROCESS_CLAHE_CLIP_LIMIT,
           gloc::GLOC_PREPROCESS_CLAHE_GRID_SIZE);
    printf("GLOC_PREPROCESS_CLARITY   : %d\n", gloc::GLOC_PREPROCESS_CLARITY);
    printf("GLOC_PREPROCESS_SHARPEN   : %d\n", gloc::GLOC_PREPROCESS_SHARPEN);
}

}; // namespace gloc