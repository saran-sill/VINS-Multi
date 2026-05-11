#include "dbow3_util.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Internal binary I/O helpers  (translation-unit scope only)
// ─────────────────────────────────────────────────────────────────────────────

namespace
{

template <typename T>
T read_bin(std::ifstream &f)
{
    T v{};
    f.read(reinterpret_cast<char *>(&v), sizeof(T));
    if (!f)
        throw std::runtime_error("Unexpected EOF while reading binary file.");
    return v;
}

} // anonymous namespace

namespace dbow3
{

static constexpr uint64_t kFeatureMagic = 0x4645415455524500ULL;
static constexpr uint64_t kFeatureVersion = 1;

static cv::Mat load_mat(std::ifstream &f)
{
    const int32_t rows = read_bin<int32_t>(f);
    const int32_t cols = read_bin<int32_t>(f);
    const int32_t type = read_bin<int32_t>(f);
    if (rows <= 0 || cols <= 0)
        return cv::Mat();
    cv::Mat m(rows, cols, type);
    const size_t row_bytes = static_cast<size_t>(cols) * m.elemSize();
    for (int r = 0; r < rows; ++r)
    {
        f.read(reinterpret_cast<char *>(m.ptr(r)),
               static_cast<std::streamsize>(row_bytes));
        if (!f)
            throw std::runtime_error("Unexpected EOF loading Mat row.");
    }
    return m;
}

// Returns true and fills feats/names if a valid cache exists, false otherwise.
static bool load_image_features(const std::string &feat_dir,
                                std::vector<ImageFeatures> &feats,
                                std::vector<std::string> &names)
{
    const std::string path = feat_dir + "/train_features.bin";
    if (!fs::exists(path))
        return false;

    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    try
    {
        const uint64_t magic = read_bin<uint64_t>(f);
        const uint64_t version = read_bin<uint64_t>(f);

        if (magic != kFeatureMagic)
        {
            std::cerr << "  [WARN] Feature cache magic mismatch — ignoring: " << path << "\n";
            return false;
        }
        if (version != kFeatureVersion)
        {
            std::cerr << "  [WARN] Feature cache version " << version
                      << " != expected " << kFeatureVersion << " — ignoring.\n";
            return false;
        }

        const uint64_t n = read_bin<uint64_t>(f);
        names.resize(n);
        feats.resize(n);

        // ── Read image names ─────────────────────────────────────────────────
        for (std::string &name : names)
        {
            const uint32_t len = read_bin<uint32_t>(f);
            name.resize(len);
            f.read(name.data(), len);
            if (!f)
                throw std::runtime_error("Unexpected EOF reading image name.");
        }

        // ── Read per-image features ──────────────────────────────────────────
        for (uint64_t i = 0; i < n; ++i)
        {
            ImageFeatures &feat = feats[i];

            feat.image_size.width = read_bin<int32_t>(f);
            feat.image_size.height = read_bin<int32_t>(f);

            const uint64_t nkp = read_bin<uint64_t>(f);
            feat.keypoints.resize(nkp);
            for (cv::KeyPoint &kp : feat.keypoints)
            {
                kp.pt.x = read_bin<float>(f);
                kp.pt.y = read_bin<float>(f);
                kp.size = read_bin<float>(f);
                kp.angle = read_bin<float>(f);
                kp.response = read_bin<float>(f);
                kp.octave = read_bin<int32_t>(f);
                kp.class_id = read_bin<int32_t>(f);
            }
            feat.orb_descriptors = load_mat(f);
            feat.beblid_descriptors = load_mat(f);

            // Guarantee row-contiguous memory for both descriptor matrices.
            // load_mat reconstructs the Mat row-by-row which can leave a
            // non-unit step[0] on some OpenCV builds, causing DBoW3 to read
            // garbage memory and segfault inside db.query() / db.add().
            if (!feat.orb_descriptors.empty() &&
                !feat.orb_descriptors.isContinuous())
                feat.orb_descriptors = feat.orb_descriptors.clone();

            if (!feat.beblid_descriptors.empty() &&
                !feat.beblid_descriptors.isContinuous())
                feat.beblid_descriptors = feat.beblid_descriptors.clone();
        }

        if (!f && !f.eof())
            throw std::runtime_error("Read error in feature cache: " + path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "  [WARN] Failed to load feature cache (" << e.what()
                  << ") — will re-extract.\n";
        feats.clear();
        names.clear();
        return false;
    }

    std::cout << "      Feature cache loaded from: " << path
              << "  (" << feats.size() << " images)\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// load_dbow3_database
//
// Steps:
//   1. Load vocabulary from vocab_path
//   2. Load feature cache from <parent(db_path)>/img_feats/train_features.bin
//   3. Load the DBoW3 database and sanity-check entry count
//
// image_folder_path (= args.train_images in the original pipeline) is the
// root folder on disk where train image files live.  It is not used during
// loading itself — it is echoed into train_image_folder_out so the caller
// can resolve img.name → full disk path without keeping track of it:
//
//   full_path = train_image_folder_out / img.name
// ─────────────────────────────────────────────────────────────────────────────

void load_dbow3_database(const std::string &db_path,
                         const std::string &vocab_path,
                         DBoW3::Database &db_out,
                         std::vector<ImageFeatures> &train_feats_out)
{
    // ── 1. Vocabulary ────────────────────────────────────────────────────────
    std::cout << "  [DBoW3] Loading vocabulary: " << vocab_path << " ...\n";

    if (!fs::exists(vocab_path))
        throw std::runtime_error("Vocabulary file not found: " + vocab_path);

    const auto t0 = std::chrono::steady_clock::now();
    DBoW3::Vocabulary voc(vocab_path);
    const auto t1 = std::chrono::steady_clock::now();

    if (voc.empty())
        throw std::runtime_error(
            "Loaded vocabulary is empty — check the file: " + vocab_path);

    std::cout << "  [DBoW3] Vocabulary loaded in "
              << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
              << "s  |  words: " << voc.size() << "\n";

    // ── 2. Feature cache ─────────────────────────────────────────────────────
    // The cache lives in a sibling folder of the DB file, named img_feats/.
    // This convention is fixed by place_recog_dbow3.
    const std::string feat_cache_dir = fs::path(db_path).parent_path().string() + "/img_feats";

    std::cout << "  [DBoW3] Loading feature cache: "
              << feat_cache_dir << "/train_features.bin ...\n";

    std::vector<std::string> cached_names; // image names embedded in the cache
    if (!load_image_features(feat_cache_dir, train_feats_out, cached_names))
        throw std::runtime_error(
            "Feature cache not found or unreadable: " +
            feat_cache_dir + "/train_features.bin");

    // ── 3. DBoW3 database ────────────────────────────────────────────────────
    std::cout << "  [DBoW3] Loading database: " << db_path << " ...\n";

    if (!fs::exists(db_path))
        throw std::runtime_error("Database file not found: " + db_path);

    db_out = DBoW3::Database(voc, false, 0);
    db_out.loadBinary(db_path);

    if (db_out.getVocabulary() == nullptr)
        throw std::runtime_error(
            "Database has no vocabulary after load — file may be corrupt.");

    if (db_out.size() == 0)
        throw std::runtime_error(
            "Database is empty after loadBinary — "
            "the binary file may be corrupt or built with a different vocabulary.");

    // Count how many train images actually have descriptors (= DB entries).
    size_t n_with_desc = 0;
    for (const auto &feat : train_feats_out)
        if (!feat.orb_descriptors.empty())
            ++n_with_desc;

    if (db_out.size() > n_with_desc)
        throw std::runtime_error(
            "DB has more entries (" + std::to_string(db_out.size()) +
            ") than train images with descriptors (" +
            std::to_string(n_with_desc) +
            "). Re-run place_recog_dbow3 to rebuild the cache and database.");

    std::cout << "  [DBoW3] Vocabulary words  : " << voc.size() << "\n"
              << "  [DBoW3] DB entries        : " << db_out.size() << "\n"
              << "  [DBoW3] Train images      : " << train_feats_out.size()
              << "  (with descriptors: " << n_with_desc << ")\n";
}

}; // namespace dbow3