#include "dbow3_util.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../vins_estimator/src/gloc/gloc.h"

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

template <typename T>
void write_bin(std::ofstream &f, const T &v)
{
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

} // anonymous namespace

namespace dbow3
{

static constexpr uint64_t kFeatureMagic = 0x4645415455524500ULL;
static constexpr uint64_t kFeatureVersion = 1;

static void save_mat(std::ofstream &f, const cv::Mat &m)
{
    const int32_t rows = m.rows, cols = m.cols, type = m.type();
    write_bin(f, rows);
    write_bin(f, cols);
    write_bin(f, type);
    if (rows > 0 && cols > 0)
    {
        const size_t row_bytes = static_cast<size_t>(cols) * m.elemSize();
        for (int r = 0; r < rows; ++r)
            f.write(reinterpret_cast<const char *>(m.ptr(r)),
                    static_cast<std::streamsize>(row_bytes));
    }
}

// Save the full train feature vector into <feat_dir>/train_features.bin
static void save_image_features(const std::vector<ImageFeatures> &feats,
                                const std::vector<std::string> &names, // ← add
                                const std::string &feat_dir)
{
    assert(feats.size() == names.size());
    fs::create_directories(feat_dir);
    const std::string path = feat_dir + "/train_features.bin";

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open feature cache for writing: " + path);

    write_bin(f, kFeatureMagic);
    write_bin(f, kFeatureVersion);

    const uint64_t n = feats.size();
    write_bin(f, n);

    // ── Write image names first ──────────────────────────────────────────────
    for (const std::string &name : names)
    {
        const uint32_t len = static_cast<uint32_t>(name.size());
        write_bin(f, len);
        f.write(name.data(), len);
    }

    // ── Write per-image features ─────────────────────────────────────────────
    for (const ImageFeatures &feat : feats)
    {
        write_bin(f, static_cast<int32_t>(feat.image_size.width));
        write_bin(f, static_cast<int32_t>(feat.image_size.height));

        const uint64_t nkp = feat.keypoints.size();
        write_bin(f, nkp);
        for (const cv::KeyPoint &kp : feat.keypoints)
        {
            write_bin(f, kp.pt.x);
            write_bin(f, kp.pt.y);
            write_bin(f, kp.size);
            write_bin(f, kp.angle);
            write_bin(f, kp.response);
            write_bin(f, static_cast<int32_t>(kp.octave));
            write_bin(f, static_cast<int32_t>(kp.class_id));
        }
        save_mat(f, feat.orb_descriptors);
        save_mat(f, feat.beblid_descriptors);
    }

    if (!f)
        throw std::runtime_error("Write error while saving feature cache: " + path);

    std::cout << "      Feature cache saved to: " << path
              << "  (" << n << " images)\n";
}

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

// ─────────────────────────────────────────────────────────────────────────────
// config_fingerprint
// ─────────────────────────────────────────────────────────────────────────────

static std::string config_fingerprint(const OrbConfig &c)
{
    std::ostringstream ss;
    ss << "nf=" << c.nfeatures
       << ",sf=" << c.scale_factor
       << ",nl=" << c.nlevels
       << ",et=" << c.edge_threshold
       << ",fl=" << c.first_level
       << ",wta=" << c.wta_k
       << ",st=" << c.score_type
       << ",ps=" << c.patch_size
       << ",ft=" << c.fast_threshold
       << ",beblid=" << (c.use_beblid ? 1 : 0)
       << ",bsf=" << c.beblid_scale
       << ",bnb=" << c.beblid_n_bits;

    const std::string s = ss.str();
    uint64_t h = 5381;
    for (unsigned char ch : s)
        h = ((h << 5) + h) ^ ch;

    std::ostringstream hex;
    hex << std::hex << std::setfill('0') << std::setw(16) << h;
    return hex.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// save_orb_config
// ─────────────────────────────────────────────────────────────────────────────

static void save_orb_config(const OrbConfig &c, const std::string &dir)
{
    const std::string path = dir + "/orb_config.txt";
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot write orb config: " + path);

    f << "# ORB extraction config — auto-generated by gloc\n"
      << "# DO NOT EDIT — changing params requires a new database.\n"
      << "nfeatures=" << c.nfeatures << "\n"
      << "scale_factor=" << c.scale_factor << "\n"
      << "nlevels=" << c.nlevels << "\n"
      << "edge_threshold=" << c.edge_threshold << "\n"
      << "first_level=" << c.first_level << "\n"
      << "wta_k=" << c.wta_k << "\n"
      << "score_type=" << c.score_type << "\n"
      << "patch_size=" << c.patch_size << "\n"
      << "fast_threshold=" << c.fast_threshold << "\n"
      << "use_beblid=" << (c.use_beblid ? 1 : 0) << "\n"
      << "beblid_scale=" << c.beblid_scale << "\n"
      << "beblid_n_bits=" << c.beblid_n_bits << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// create_dbow3_database
//
// The caller owns and passes the already-constructed extractors so this
// function reuses exactly the same orb_extractor_ / beblid_extractor_ that
// Gloc uses for query images — guaranteeing descriptor compatibility.
//
// cfg is passed separately for cache fingerprinting and orb_config.txt only;
// it does NOT drive extractor construction here.
//
// Caller in gloc.cpp:
//   dbow3::OrbConfig cfg;
//   cfg.nfeatures = GLOC_ORB_NFEATURES; ... (fill from GLOC_* params)
//   dbow3::create_dbow3_database(map_.images, GLOC_COLMAP_IMG_FOLDER,
//       GLOC_DBOW3_VOCAB, GLOC_DBOW3_AUTO_CREATED_DB_FOLDER,
//       cfg, orb_extractor_.get(), beblid_extractor_,
//       map_.db, map_.feats);
// ─────────────────────────────────────────────────────────────────────────────

void create_dbow3_database(const std::vector<colmap::Image> &map_images,
                           const std::string &img_folder,
                           const std::string &vocab_path,
                           const std::string &auto_db_root,
                           const OrbConfig &cfg,
                           PointFeatureExtractor *orb_extractor,
                           cv::Ptr<cv::xfeatures2d::BEBLID> beblid_extractor,
                           DBoW3::Database &db_out,
                           std::vector<ImageFeatures> &train_feats_out)
{
    if (!orb_extractor)
        throw std::runtime_error("[createDB] orb_extractor is null");

    // ── Determine cache directory from config fingerprint ─────────────────────
    const std::string fingerprint = config_fingerprint(cfg);
    const std::string cache_dir = auto_db_root + "/" + fingerprint;
    const std::string db_path = cache_dir + "/database.dbow3";
    const std::string feat_path = cache_dir + "/train_features.bin";

    printf("  [INFO] [createDB] auto_db_root : %s\n", auto_db_root.c_str());
    printf("  [INFO] [createDB] fingerprint  : %s\n", fingerprint.c_str());
    printf("  [INFO] [createDB] cache_dir    : %s\n", cache_dir.c_str());

    // ── Try loading existing cache ────────────────────────────────────────────
    if (fs::exists(db_path) && fs::exists(feat_path))
    {
        printf("  [INFO] [createDB] Found existing cache — loading ...\n");
        try
        {
            load_dbow3_database(db_path, vocab_path, db_out, train_feats_out);
            printf("  [INFO] [createDB] Cache loaded OK (db=%zu feats=%zu).\n", db_out.size(), train_feats_out.size());
            return;
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "  [WARN] [createDB] Cache load failed (%s) — re-extracting.\n", e.what());
        }
    }

    // ── Load vocabulary ───────────────────────────────────────────────────────
    printf("  [INFO] [createDB] Loading vocabulary: %s\n", vocab_path.c_str());
    if (!fs::exists(vocab_path))
        throw std::runtime_error("Vocabulary not found: " + vocab_path);

    DBoW3::Vocabulary vocab(vocab_path);
    if (vocab.empty())
        throw std::runtime_error("Loaded vocabulary is empty: " + vocab_path);

    printf("  [INFO] [createDB] Vocabulary loaded (%zu words).\n", vocab.size());
    if (beblid_extractor)
        printf("  [INFO] [createDB] BEBLID extractor provided — will extract BEBLID descriptors.\n");

    // ── Extract features using the caller's extractor instances ──────────────
    const std::size_t N = map_images.size();
    train_feats_out.resize(N);
    std::vector<std::string> names(N);
    std::size_t n_ok = 0, n_fail = 0;

    printf("  [INFO] [createDB] Extracting features from %zu images ...\n", N);

    for (std::size_t ti = 0; ti < N; ++ti)
    {
        names[ti] = map_images[ti].name;
        const std::string img_path = img_folder + "/" + map_images[ti].name;

        cv::Mat gray = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (gray.empty())
        {
            fprintf(stderr, "  [WARN] [createDB] cannot read image: %s — skipping.\n", img_path.c_str());
            ++n_fail;
            continue;
        }

        ImageFeatures &feat = train_feats_out[ti];
        feat.image_size = gray.size();

        // Use the same extractor instance as query — identical params guaranteed.
        orb_extractor->extract(gray,
                               feat.keypoints,
                               &feat.orb_descriptors);

        if (!feat.orb_descriptors.empty() && !feat.orb_descriptors.isContinuous())
            feat.orb_descriptors = feat.orb_descriptors.clone();

        if (beblid_extractor && !feat.keypoints.empty())
        {
            beblid_extractor->compute(gray, feat.keypoints, feat.beblid_descriptors);
            if (!feat.beblid_descriptors.empty() &&
                !feat.beblid_descriptors.isContinuous())
                feat.beblid_descriptors = feat.beblid_descriptors.clone();
        }

        ++n_ok;
        if (ti % 200 == 0 || ti == N - 1)
            printf("  [INFO] [createDB] %zu/%zu  ok=%zu  fail=%zu\n", ti + 1, N, n_ok, n_fail);
    }

    printf("  [INFO] [createDB] Extraction done: ok=%zu  fail=%zu.\n", n_ok, n_fail);
    if (n_ok == 0)
        throw std::runtime_error(
            "[createDB] All images failed to load — check img_folder: " + img_folder);

    // ── Build DBoW3 database ──────────────────────────────────────────────────
    printf("  [INFO] [createDB] Building DBoW3 database ...\n");
    db_out = DBoW3::Database(vocab, /*use_di=*/false, /*di_levels=*/0);
    for (std::size_t ti = 0; ti < N; ++ti)
    {
        if (!train_feats_out[ti].orb_descriptors.empty())
            db_out.add(train_feats_out[ti].orb_descriptors);
        else
            db_out.add(cv::Mat()); // empty entry keeps index aligned with map_images
    }
    printf("  [INFO] [createDB] Database built: %zu entries.\n", db_out.size());

    // ── Save to disk ──────────────────────────────────────────────────────────
    printf("  [INFO] [createDB] Saving cache to: %s\n", cache_dir.c_str());
    fs::create_directories(cache_dir);

    save_orb_config(cfg, cache_dir);
    save_image_features(train_feats_out, names, cache_dir);
    db_out.saveBinary(db_path);

    printf("  [INFO] [createDB] Saved. Next startup will load from cache (fingerprint=%s).\n", fingerprint.c_str());
}

}; // namespace dbow3