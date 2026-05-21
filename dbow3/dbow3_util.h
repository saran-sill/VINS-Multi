#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../colmap/colmap_util.h"
#include "../point_features/point_features.h"
#include "DBoW3/DBoW3.h"

namespace dbow3
{

// ─────────────────────────────────────────────────────────────────────────────
// Per-image feature store
// ─────────────────────────────────────────────────────────────────────────────

struct ImageFeatures
{
    cv::Size image_size;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat orb_descriptors;    // CV_8U — used for DBoW3 (vocab is ORB-trained)
    cv::Mat beblid_descriptors; // CV_8U — populated only when use_beblid=true
};

struct OrbConfig
{
    int nfeatures = 1000;
    float scale_factor = 1.2f;
    int nlevels = 8;
    int edge_threshold = 31;
    int first_level = 0;
    int wta_k = 2;
    int score_type = 0;
    int patch_size = 31;
    int fast_threshold = 20;
    bool use_beblid = false;
    float beblid_scale = 1.0f;
    int beblid_n_bits = 256;
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API — DBoW3 database loader
// ─────────────────────────────────────────────────────────────────────────────
void create_dbow3_database(const std::vector<colmap::Image> &map_images,
                           const std::string &img_folder,
                           const std::string &vocab_path,
                           const std::string &auto_db_root,
                           const OrbConfig &cfg,
                           PointFeatureExtractor *orb_extractor,
                           cv::Ptr<cv::xfeatures2d::BEBLID> beblid_extractor,
                           DBoW3::Database &db_out,
                           std::vector<ImageFeatures> &train_feats_out);

/**
 * Load the DBoW3 vocabulary, feature cache, and database in one call.
 *
 * Expected directory layout (produced by place_recog_dbow3):
 *
 *   <db_path>               ← e.g. /data/run/database_orb.bin
 *   <db_path>/../img_feats/
 *       train_features.bin  ← feature cache written by place_recog_dbow3
 *
 * Steps performed:
 *   1. Load DBoW3::Vocabulary from vocab_path.
 *   2. Load the feature cache from <parent(db_path)>/img_feats/train_features.bin.
 *      The image_folder_path is stored alongside the features so the caller can
 *      resolve full disk paths for each train image via img.name (same role as
 *      args.train_images in the original pipeline).
 *   3. Load the DBoW3::Database from db_path and sanity-check it.
 *
 * @param db_path            Full path to the DBoW3 binary database file.
 * @param vocab_path         Full path to the ORB vocabulary file (.dbow3/.bin).
 * @param db_out             Output: loaded and validated DBoW3 database.
 * @param train_feats_out    Output: feature cache ordered to match the cache file.
 *
 * @throws std::runtime_error if any file is missing, unreadable, empty, or
 *         if the database has more entries than images with descriptors.
 */
void load_dbow3_database(const std::string &db_path,
                         const std::string &vocab_path,
                         DBoW3::Database &db_out,
                         std::vector<ImageFeatures> &train_feats_out);

}; // namespace dbow3