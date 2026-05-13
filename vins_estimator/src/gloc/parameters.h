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

extern int GLOC_ENABLED;
extern std::string GLOC_COLMAP_FOLDER;
extern std::string GLOC_COLMAP_IMG_FOLDER;
extern std::string GLOC_DBOW3_DATABASE;
extern std::string GLOC_DBOW3_VOCAB;

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
extern int GLOC_USE_BEBLID;
extern float GLOC_BEBLID_SCALE_FACTOR;
extern int GLOC_BEBLID_N_BITS; // 256 or 512

// ── Feature matching ─────────────────────────────────────────────────────────

// 0 = brute-force robustMatch, 1 = GMS matchGMS.
extern int GLOC_USE_GMS;
extern float GLOC_GMS_THRESHOLD;
extern int GLOC_GMS_WITH_ROTATION;
extern int GLOC_GMS_WITH_SCALE;

extern float GLOC_MATCH_LOWE_RATIO;
extern int GLOC_MATCH_MAX_DIST;
extern float GLOC_MATCH_GEOM_REPROJ_TH;
extern float GLOC_MATCH_GEOM_CONFIDENCE;
extern float GLOC_MATCH_GEOM_SAMPSON_SQ;
extern int GLOC_MATCH_MIN_INLIERS;

void readParameters(std::string config_file);

}; // namespace gloc