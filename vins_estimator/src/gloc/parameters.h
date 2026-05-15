#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <vector>

namespace gloc
{

extern bool GLOC_ENABLED;
extern std::string GLOC_COLMAP_SPARSE_FOLDER;
extern std::string GLOC_COLMAP_IMG_FOLDER;
extern std::string GLOC_DBOW3_DATABASE;
extern std::string GLOC_DBOW3_VOCAB;

// Optional: explicit path to world_transform.txt.
// When empty, gloc falls back to looking for the file inside
// GLOC_COLMAP_SPARSE_FOLDER. Set via gloc_world_tmat_file in yaml.
extern std::string GLOC_WORLD_TMAT_FILE;

// ── DBoW3 query ──────────────────────────────────────────────────────────────

// Number of top candidates returned by DBoW3 per query image (N in the plan).
extern int GLOC_DBOW3_MAX_RESULTS;

// Minimum DBoW3 score to retain a candidate.
extern double GLOC_DBOW3_MIN_SCORE;

// ── Consensus voting ─────────────────────────────────────────────────────────

// Tolerance (metres) on |world_dist - local_dist| for two candidates across
// different keyframes to be considered "agreeing".
extern double GLOC_VOTE_EPS_M;

// Minimum number of other keyframes whose candidate agrees with this one
// before it is accepted.  When set to -1 the code uses ceil((X-1)/2) where
// X is the current working-set size (majority rule).
extern int GLOC_VOTE_MIN_VOTES;

// ── ORB feature extraction ───────────────────────────────────────────────────

extern int GLOC_ORB_NFEATURES;
extern float GLOC_ORB_SCALE_FACTOR;
extern int GLOC_ORB_NLEVELS;
extern int GLOC_ORB_EDGE_THRESHOLD;
extern int GLOC_ORB_FIRST_LEVEL;
extern int GLOC_ORB_WTA_K;
extern int GLOC_ORB_SCORE_TYPE; // 0 = HARRIS, 1 = FAST
extern int GLOC_ORB_PATCH_SIZE;
extern int GLOC_ORB_FAST_THRESHOLD;

// ── Descriptor choice ────────────────────────────────────────────────────────

// 0 = use ORB descriptors, 1 = compute and use BEBLID descriptors.
extern bool GLOC_USE_BEBLID;
extern float GLOC_BEBLID_SCALE_FACTOR;
extern int GLOC_BEBLID_N_BITS; // 256 or 512

// ── Feature matching ─────────────────────────────────────────────────────────

// 0 = brute-force robustMatch, 1 = GMS matchGMS.
extern bool GLOC_USE_GMS;
extern float GLOC_GMS_THRESHOLD;
extern bool GLOC_GMS_WITH_ROTATION;
extern bool GLOC_GMS_WITH_SCALE;

extern float GLOC_MATCH_LOWE_RATIO;
extern int GLOC_MATCH_MAX_DIST;
extern float GLOC_MATCH_GEOM_REPROJ_TH;
extern float GLOC_MATCH_GEOM_CONFIDENCE;
extern float GLOC_MATCH_GEOM_SAMPSON_SQ;
extern int GLOC_MATCH_MIN_INLIERS;
extern float GLOC_SUBSAMPLE_MIN_DIST_PX; // 0 = disabled

void readParameters(std::string config_file);

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — Ceres optimization
// ─────────────────────────────────────────────────────────────────────────────

// Minimum number of valid keyframe-slots (pipeline_done && best_train_idx>=0)
// required before the optimizer is called.
// GLOC_MIN_PAIRS_FIRST_SNAP is used before the first successful snap (stricter).
// GLOC_MIN_PAIRS is used once snapped (more relaxed).
extern int GLOC_MIN_PAIRS;
extern int GLOC_MIN_PAIRS_FIRST_SNAP;

// Relative-pose chain half-width: keyframe i is connected to i+1 .. i+k_rel.
// 1 = adjacent only, 2 = i+1 and i+2, etc.
extern int GLOC_REL_POSE_K;

// Weights for each energy term.
extern double GLOC_W_EPIPOLAR;    // Sampson epipolar
extern double GLOC_W_REPROJ;      // inverse-depth reprojection
extern double GLOC_W_REL_POSE;    // relative local pose
extern double GLOC_W_WORLD_PRIOR; // world prior (only when snapped)

// Huber loss delta (virtual pixels) applied to epipolar and reprojection terms.
extern double GLOC_HUBER_DELTA;

// Reprojection inlier threshold (virtual pixels) used for post-solve inlier
// counting and for deciding whether to accept the solution.
extern double GLOC_INLIER_THRESH_PX;

// Solver iteration limits.
extern int GLOC_MAX_ITERS;
extern int GLOC_INIT_ITERS; // fast burn-in pass before main solve

// Minimum inlier fraction required to accept the solution and set snapped.
extern double GLOC_MIN_INLIER_RATIO;

// Maximum depth (metres) for inverse-depth lower bound: rho >= 1/GLOC_MAX_DEPTH_M.
extern double GLOC_MAX_DEPTH_M;

// When true, optimization is constrained to 4-DOF (x, y, z, yaw) by fixing
// pitch and roll. Pitch and roll are trusted from IMU integration. When false
// (default), full 6-DOF optimization is used.
extern bool GLOC_USE_4DOF;

// When true, relative poses between keyframes are fixed (trusted from VINS).
// The only optimization variables are T_map_local (4 or 6 DOF depending on
// GLOC_USE_4DOF). When false (default), per-keyframe poses are optimized
// independently with soft relative-pose constraints.
extern bool GLOC_FIX_REL_POSES;

// Optional: folder prefix for debug images saved by runCorrespondences.
// Empty = disabled. Example: "/mnt/EXDISK/tmp/dbg_"
// Images are saved as:
//   <prefix><t_kf>_g<g>_query.jpg       — query image with keypoints
//   <prefix><t_kf>_g<g>_match.jpg       — side-by-side with match lines
extern std::string GLOC_DEBUG_FOLDER;

// ─────────────────────────────────────────────────────────────────────────────
// Image preprocessing (applied to query images before ORB extraction)
// ─────────────────────────────────────────────────────────────────────────────

extern bool GLOC_PREPROCESS_WHITE_BALANCE;

extern bool GLOC_PREPROCESS_DENOISE;
extern int GLOC_PREPROCESS_DENOISE_D;
extern double GLOC_PREPROCESS_DENOISE_SIGMA_COLOR;
extern double GLOC_PREPROCESS_DENOISE_SIGMA_SPACE;

extern bool GLOC_PREPROCESS_DENOISE_COLOR;
extern double GLOC_PREPROCESS_DENOISE_H_LUMINANCE;
extern double GLOC_PREPROCESS_DENOISE_H_COLOR;

extern bool GLOC_PREPROCESS_GAMMA;
extern double GLOC_PREPROCESS_GAMMA_VALUE;

extern bool GLOC_PREPROCESS_TONEMAP;
extern double GLOC_PREPROCESS_TONEMAP_GAMMA;
extern double GLOC_PREPROCESS_TONEMAP_HIGHLIGHT;
extern double GLOC_PREPROCESS_TONEMAP_SHADOW_LIFT;

extern bool GLOC_PREPROCESS_CLAHE;
extern double GLOC_PREPROCESS_CLAHE_CLIP_LIMIT;
extern int GLOC_PREPROCESS_CLAHE_GRID_SIZE;

extern bool GLOC_PREPROCESS_CLARITY;
extern double GLOC_PREPROCESS_CLARITY_AMOUNT;
extern double GLOC_PREPROCESS_CLARITY_SIGMA;

extern bool GLOC_PREPROCESS_SHARPEN;
extern double GLOC_PREPROCESS_SHARPEN_SIGMA;
extern double GLOC_PREPROCESS_SHARPEN_AMOUNT;

}; // namespace gloc