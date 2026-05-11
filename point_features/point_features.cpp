#include "point_features.h"

//-----------------------------------------------------------------
// PointFeatureExtractor
//-----------------------------------------------------------------
PointFeatureExtractor::PointFeatureExtractor()
{

}

PointFeatureExtractor::~PointFeatureExtractor()
{

}

void PointFeatureExtractor::drawKeypoints(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & outImage)
{
    cv::drawKeypoints(image, keyPoints, outImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

//-----------------------------------------------------------------
// PointFeatureExtractorSift
//-----------------------------------------------------------------
PointFeatureExtractorSift::PointFeatureExtractorSift(const Parameters & params)
    : PointFeatureExtractor()
{
    setParams(params);
}

PointFeatureExtractorSift::~PointFeatureExtractorSift()
{

}

void PointFeatureExtractorSift::setParams(const Parameters & params)
{
    m_detector = cv::SIFT::create(params.nfeatures, params.nOctaveLayers, params.contrastThreshold, params.edgeThreshold, params.sigma, params.descriptorType);
    m_extractor = m_detector;
}

bool PointFeatureExtractorSift::extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor)
{
    keyPoints.clear();

    const int num_channels = image.channels();

    cv::Mat input_img = image;

    // if (num_channels == 1)
    //     input_img = image;
    // else if (num_channels == 3)
    // {
    //     cv::cvtColor(image, input_img, cv::COLOR_BGR2GRAY);
    // }
    // else return false;

    m_detector->detect(input_img, keyPoints);

    if (descriptor != nullptr)
    {
        m_extractor->compute(input_img, keyPoints, *descriptor);
    }

    return true;
}

bool PointFeatureExtractorSift::computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor)
{
    cv::Mat input_img = image;

    m_extractor->compute(input_img, keyPoints, descriptor);

    return true;
}

//-----------------------------------------------------------------
// PointFeatureExtractorSift
//-----------------------------------------------------------------
PointFeatureExtractorSurf::PointFeatureExtractorSurf(const Parameters & params)
    : PointFeatureExtractor()
{
    setParams(params);
}

PointFeatureExtractorSurf::~PointFeatureExtractorSurf()
{

}

void PointFeatureExtractorSurf::setParams(const Parameters & params)
{
    m_detector = cv::xfeatures2d::SURF::create(params.hessianThreshold, params.nOctaves, params.nOctaveLayers, params.extended, params.upright);
    m_extractor = m_detector;
}

bool PointFeatureExtractorSurf::extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor)
{
    keyPoints.clear();

    const int num_channels = image.channels();

    cv::Mat input_img = image;

    // if (num_channels == 1)
    //     input_img = image;
    // else if (num_channels == 3)
    // {
    //     cv::cvtColor(image, input_img, cv::COLOR_BGR2GRAY);
    // }
    // else return false;

    m_detector->detect(input_img, keyPoints);

    if (descriptor != nullptr)
    {
        m_extractor->compute(input_img, keyPoints, *descriptor);
    }

    return true;
}

bool PointFeatureExtractorSurf::computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor)
{
    cv::Mat input_img = image;

    m_extractor->compute(input_img, keyPoints, descriptor);

    return true;
}

//-----------------------------------------------------------------
// PointFeatureExtractorORB
//-----------------------------------------------------------------
PointFeatureExtractorORB::PointFeatureExtractorORB(const Parameters & _params)
    : PointFeatureExtractor(), params(_params)
{
    setParams(_params);
}

PointFeatureExtractorORB::~PointFeatureExtractorORB()
{

}

void PointFeatureExtractorORB::setParams(const Parameters & _params)
{
    params = _params;
    m_detector = cv::ORB::create(params.nfeatures,
                                 params.scaleFactor,
                                 params.nlevels,
                                 params.edgeThreshold,
                                 params.firstLevel,
                                 params.WTA_K,
                                 (cv::ORB::ScoreType)params.scoreType,
                                 params.patchSize,
                                 params.fastThreshold);
    m_extractor = m_detector;
}

bool PointFeatureExtractorORB::extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor)
{
    keyPoints.clear();

    const int num_channels = image.channels();

    cv::Mat input_img = image;

    // if (num_channels == 1)
    //     input_img = image;
    // else if (num_channels == 3)
    // {
    //     cv::cvtColor(image, input_img, cv::COLOR_BGR2GRAY);
    // }
    // else return false;

    m_detector->detect(input_img, keyPoints);

    // Reject keypoints whose size is below the threshold
    if (params.keypoint_min_size_pixels > 0.0f)
    {
        keyPoints.erase(
            std::remove_if(keyPoints.begin(), keyPoints.end(),
                           [this](const cv::KeyPoint &kp) {
                               return kp.size < params.keypoint_min_size_pixels;
                           }),
            keyPoints.end());
    }

    if (descriptor != nullptr)
    {
        m_extractor->compute(input_img, keyPoints, *descriptor);

        // if (descriptor->type() != CV_32F)
        // {
        //     descriptor->convertTo(*descriptor, CV_32F);
        // }
    }

    return true;
}

bool PointFeatureExtractorORB::computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor)
{
    cv::Mat input_img = image;

    m_extractor->compute(input_img, keyPoints, descriptor);

    if (descriptor.type() != CV_32F)
    {
        descriptor.convertTo(descriptor, CV_32F);
    }

    return true;
}

//-----------------------------------------------------------------
// GridingFeatureExtractorORB
//-----------------------------------------------------------------
GridingFeatureExtractorORB::GridingFeatureExtractorORB(const int grid_x_, const int grid_y_, const Parameters & _params)
    : grid_x(grid_x_)
    , grid_y(grid_y_)
    , PointFeatureExtractor()
    , params(_params)
{
    setParams(_params);
}

GridingFeatureExtractorORB::~GridingFeatureExtractorORB()
{

}

void GridingFeatureExtractorORB::setParams(const Parameters & _params)
{
    params = _params;
    const int num_features = params.nfeatures;
    if (num_features < grid_x * grid_y)
    {
        double ratio = (double)grid_x / (double)grid_y;
        grid_y = std::ceil(std::sqrt(num_features / ratio));
        grid_x = std::ceil(grid_y * ratio);
    }
    num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;

    m_detector = cv::ORB::create(num_features_grid,
                                 params.scaleFactor,
                                 params.nlevels,
                                 params.edgeThreshold,
                                 params.firstLevel,
                                 params.WTA_K,
                                 (cv::ORB::ScoreType)params.scoreType,
                                 params.patchSize,
                                 params.fastThreshold);
    m_extractor = m_detector;
}

bool GridingFeatureExtractorORB::extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor)
{
    keyPoints.clear();

    if (image.empty() || grid_x <= 0 || grid_y <= 0)
        return false;

    // We want to have equally distributed features
    // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
    // NOTE: Thus if we extract 1 point per grid we have
    // NOTE:    -> 1 = num_features / (grid_x * grid_y))
    // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
    // NOTE:    -> grid_y = sqrt(num_features / ratio)

    // Calculate the size our extraction boxes should be
    int size_x = image.cols / grid_x;
    int size_y = image.rows / grid_y;

    if (size_x <= 0 || size_y <= 0)
        return false;

    // Parallelize our 2d grid extraction!!
    int ct_cols = image.cols / size_x;
    int ct_rows = image.rows / size_y;
    std::vector<std::vector<cv::KeyPoint>> collection(ct_cols * ct_rows);

    for (int r = 0; r < ct_cols * ct_rows; ++r)
    {
        // Calculate what cell xy value we are in
        int x = r % ct_cols * size_x;
        int y = r / ct_cols * size_y;

        // Skip if we are out of bounds
        if (x + size_x > image.cols || y + size_y > image.rows)
            continue;

        // Calculate where we should be extracting from
        cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

        std::vector<cv::KeyPoint> pts_new;
        cv::Mat roi = image(img_roi);
        m_detector->detect(roi, pts_new);

        // Reject keypoints whose size is below the threshold
        if (params.keypoint_min_size_pixels > 0.0f)
        {
            pts_new.erase(
                std::remove_if(pts_new.begin(), pts_new.end(),
                               [this](const cv::KeyPoint &kp) {
                                   return kp.size < params.keypoint_min_size_pixels;
                               }),
                pts_new.end());
        }

        // Now lets get the top number from this
        std::sort(pts_new.begin(), pts_new.end(), [](const cv::KeyPoint &first, const cv::KeyPoint &second){ return first.response > second.response; });

        // Append the "best" ones to our vector
        // Note that we need to "correct" the point u,v since we extracted it in a ROI
        // So we should append the location of that ROI in the image
        for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++)
        {
            // Create keypoint
            cv::KeyPoint pt_cor = pts_new.at(i);
            pt_cor.pt.x += (float)x;
            pt_cor.pt.y += (float)y;

            // Reject if out of bounds (shouldn't be possible...)
            if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x >= image.cols || (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y >= image.rows)
                continue;

            // Check if it is in the mask region
            // NOTE: mask has max value of 255 (white) if it should be removed
            // if (!mask.empty() && mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127)
            //     continue;
            collection.at(r).push_back(pt_cor);
        }
    }

    // Combine all the collections into our single vector
    for (size_t r = 0; r < collection.size(); r++)
    {
        keyPoints.insert(keyPoints.end(), collection.at(r).begin(), collection.at(r).end());
    }

    // Return if no points
    // printf("#kps : %ld\n", keyPoints.size());
    if (keyPoints.empty())
        return false;

    // Compute descriptors
    if (descriptor != nullptr)
    {
        m_extractor->compute(image, keyPoints, *descriptor);
    }

    return true;
}

bool GridingFeatureExtractorORB::computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor)
{
    cv::Mat input_img = image;

    m_extractor->compute(input_img, keyPoints, descriptor);

    return true;
}


//-----------------------------------------------------------------
// GridingFeatureExtractorFASTBrief
//-----------------------------------------------------------------
GridingFeatureExtractorFASTBrief::GridingFeatureExtractorFASTBrief(const int num_features_, const int grid_x_, const int grid_y_, const int threshold_, const bool nonmaxSuppression_, const bool doCornerSubpix_)
    : num_features(num_features_)
    , grid_x(grid_x_)
    , grid_y(grid_y_)
    , threshold(threshold_)
    , nonmaxSuppression(nonmaxSuppression_)
    , doCornerSubpix(doCornerSubpix_)
{
    feat_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
}

GridingFeatureExtractorFASTBrief::~GridingFeatureExtractorFASTBrief()
{

}

bool GridingFeatureExtractorFASTBrief::extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor, const cv::Mat & mask)
{
    keyPoints.clear();

    if (image.empty() || grid_x <= 0 || grid_y <= 0)
        return false;

    // We want to have equally distributed features
    // NOTE: If we have more grids than number of total points, we calc the biggest grid we can do
    // NOTE: Thus if we extract 1 point per grid we have
    // NOTE:    -> 1 = num_features / (grid_x * grid_y))
    // NOTE:    -> grid_x = ratio * grid_y (keep the original grid ratio)
    // NOTE:    -> grid_y = sqrt(num_features / ratio)
    if (num_features < grid_x * grid_y)
    {
        double ratio = (double)grid_x / (double)grid_y;
        grid_y = std::ceil(std::sqrt(num_features / ratio));
        grid_x = std::ceil(grid_y * ratio);
    }
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    assert(grid_x > 0);
    assert(grid_y > 0);
    assert(num_features_grid > 0);

    // Calculate the size our extraction boxes should be
    int size_x = image.cols / grid_x;
    int size_y = image.rows / grid_y;

    if (size_x <= 0 || size_y <= 0)
        return false;

    // Make sure our sizes are not zero
    assert(size_x > 0);
    assert(size_y > 0);

    // Gray image
    cv::Mat image_gray;
    if (image.channels() != 1)
    {
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    }
    else image_gray = image.clone();

    // Parallelize our 2d grid extraction!!
    int ct_cols = image.cols / size_x;
    int ct_rows = image.rows / size_y;
    std::vector<std::vector<cv::KeyPoint>> collection(ct_cols * ct_rows);

    for (int r = 0; r < ct_cols * ct_rows; ++r)
    {
        // Calculate what cell xy value we are in
        int x = r % ct_cols * size_x;
        int y = r / ct_cols * size_y;

        // Skip if we are out of bounds
        if (x + size_x > image.cols || y + size_y > image.rows)
            continue;

        // Calculate where we should be extracting from
        cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

        // Extract FAST features for this part of the image
        std::vector<cv::KeyPoint> pts_new;
        cv::FAST(image_gray(img_roi), pts_new, threshold, nonmaxSuppression);

        // feat_extractor->extract(image(img_roi), pts_new, nullptr);

        // printf("#pts_new : %ld\n", pts_new.size());

        // Now lets get the top number from this
        std::sort(pts_new.begin(), pts_new.end(), [](cv::KeyPoint first, cv::KeyPoint second){ return first.response > second.response; });

        // Append the "best" ones to our vector
        // Note that we need to "correct" the point u,v since we extracted it in a ROI
        // So we should append the location of that ROI in the image
        for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++)
        {
            // Create keypoint
            cv::KeyPoint pt_cor = pts_new.at(i);
            pt_cor.pt.x += (float)x;
            pt_cor.pt.y += (float)y;

            // Reject if out of bounds (shouldn't be possible...)
            if ((int)pt_cor.pt.x < 0 || (int)pt_cor.pt.x >= image.cols || (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y >= image.rows)
                continue;

            // Check if it is in the mask region
            // NOTE: mask has max value of 255 (white) if it should be removed
            if (!mask.empty() && mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127)
                continue;
            collection.at(r).push_back(pt_cor);
        }
    }

    // Combine all the collections into our single vector
    for (size_t r = 0; r < collection.size(); r++)
    {
        keyPoints.insert(keyPoints.end(), collection.at(r).begin(), collection.at(r).end());
    }

    // Return if no points
    // printf("#kps : %ld\n", keyPoints.size());
    if (keyPoints.empty())
        return false;

    // Sub-pixel refinement parameters
    if (doCornerSubpix)
    {
        cv::Size win_size = cv::Size(5, 5);
        cv::Size zero_zone = cv::Size(-1, -1);
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

        // Get vector of points
        std::vector<cv::Point2f> pts_refined;
        for (size_t i = 0; i < keyPoints.size(); i++)
        {
            pts_refined.push_back(keyPoints.at(i).pt);
        }

        // Finally get sub-pixel for all extracted features
        cv::cornerSubPix(image_gray, pts_refined, win_size, zero_zone, term_crit);

        // Save the refined points!
        for (size_t i = 0; i < keyPoints.size(); i++)
        {
            keyPoints.at(i).pt = pts_refined.at(i);
        }
    }

    // Compute descriptors
    if (descriptor != nullptr)
    {
        feat_extractor->compute(image, keyPoints, *descriptor);
    }

    return true;
}

//-----------------------------------------------------------------
// PointFeatureMatcher
//-----------------------------------------------------------------
PointFeatureMatcher::PointFeatureMatcher()
{

}

PointFeatureMatcher::~PointFeatureMatcher()
{

}

int PointFeatureMatcher::ratio_test(std::vector<std::vector<cv::DMatch>> &matches, const float &lowe_ratio)
{
    int removed = 0;

    // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator= matches.begin(); matchIterator!= matches.end(); ++matchIterator)
    {
        // if 2 NN has been identified
        if (matchIterator->size() > 1)
        {
            // check distance ratio
            if ((*matchIterator)[0].distance > lowe_ratio * ((*matchIterator)[1].distance))
            {
                matchIterator->clear(); // remove match
                removed++;
            }
        }
        else
        {
            // does not have 2 neighbours
            matchIterator->clear(); // remove match
            removed++;
        }
    }

    return removed;
}

int PointFeatureMatcher::ratio_test(std::vector<std::vector<cv::DMatch>> &matches, const float &lowe_ratio, const int &max_dist)
{
    int removed = 0;

    for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
    {
        if (matchIterator->size() > 1)
        {
            // check distance ratio and absolute distance cap
            if ((*matchIterator)[0].distance > lowe_ratio * (*matchIterator)[1].distance ||
                (*matchIterator)[0].distance > max_dist)
            {
                matchIterator->clear();
                removed++;
            }
        }
        else
        {
            matchIterator->clear();
            removed++;
        }
    }

    return removed;
}

void PointFeatureMatcher::symmetry_test(const std::vector<std::vector<cv::DMatch>> &matches01,
                                              const std::vector<std::vector<cv::DMatch>> &matches10,
                                              std::vector<cv::DMatch> &symMatches01)
{
    symMatches01.clear();

    // for all matches image 0 -> image 1
    for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches01.begin(); matchIterator1 != matches01.end(); ++matchIterator1)
    {
        // ignore deleted matches
        if (matchIterator1->empty() || matchIterator1->size() < 2)
            continue;

        // for all matches image 1 -> image 0
        for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 = matches10.begin(); matchIterator2 != matches10.end(); ++matchIterator2)
        {
            // ignore deleted matches
            if (matchIterator2->empty() || matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
            {
                // add symmetrical match
                symMatches01.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
                                                  (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
            }
        }
    }
}

void PointFeatureMatcher::drawMatches(const cv::Mat & image0, const std::vector<cv::KeyPoint> &keyPoints0,
                            const cv::Mat & image1, const std::vector<cv::KeyPoint> &keyPoints1,
                            const std::vector<cv::DMatch>& matches,
                            cv::Mat & outImage,
                            const int thickness)
{
    cv::drawMatches(image0, keyPoints0, image1, keyPoints1, matches, outImage, thickness,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT
                    // | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
                    | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                    );
}

void PointFeatureMatcher::matchesToPointCorrespondences(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                                        const std::vector<cv::DMatch> &matches, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &pt_correspondences,
                                                        const double & point_scale_factor)
{
    const size_t num_matches = matches.size();

    pt_correspondences.resize(num_matches);

    for (unsigned int match_index = 0; match_index < matches.size(); ++match_index)
    {
        const cv::Point2f pt0 = keypoints0[matches[match_index].queryIdx].pt;
        const cv::Point2f pt1 = keypoints1[matches[match_index].trainIdx].pt;

        pt_correspondences[match_index] = std::make_pair(Eigen::Vector2d(pt0.x, pt0.y) * point_scale_factor, Eigen::Vector2d(pt1.x, pt1.y) * point_scale_factor);
    }
}

void PointFeatureMatcher::matchesToPointCorrespondences(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                                        const std::vector<cv::DMatch> &matches, std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> &pt_correspondences,
                                                        const float &point_scale_factor)
{
    const size_t num_matches = matches.size();

    pt_correspondences.resize(num_matches);

    for (unsigned int match_index = 0; match_index < matches.size(); ++match_index)
    {
        const cv::Point2f pt0 = keypoints0[matches[match_index].queryIdx].pt;
        const cv::Point2f pt1 = keypoints1[matches[match_index].trainIdx].pt;

        pt_correspondences[match_index] = std::make_pair(Eigen::Vector2f(pt0.x, pt0.y) * point_scale_factor, Eigen::Vector2f(pt1.x, pt1.y) * point_scale_factor);
    }
}

void PointFeatureMatcher::matchesToPoints(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                          const std::vector<cv::DMatch> &matches,
                                          std::vector<cv::Point2f> &points1,
                                          std::vector<cv::Point2f> &points2,
                                          const float &point_scale_factor)
{
    const size_t num_matches = matches.size();

    points1.resize(num_matches);
    points2.resize(num_matches);

    for (unsigned int match_index = 0; match_index < matches.size(); ++match_index)
    {
        points1[match_index] = keypoints0[matches[match_index].queryIdx].pt * point_scale_factor;
        points2[match_index] = keypoints1[matches[match_index].trainIdx].pt * point_scale_factor;

    }
}

void PointFeatureMatcher::rowAlignTest(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                       std::vector<cv::DMatch> &matches, const float row_align_pixel_threshold)
{
    const size_t num_matches = matches.size();

    std::vector<cv::DMatch> new_matches;

    for (unsigned int match_index = 0; match_index < matches.size(); ++match_index)
    {
        const cv::Point2f pt0 = keypoints0[matches[match_index].queryIdx].pt;
        const cv::Point2f pt1 = keypoints1[matches[match_index].trainIdx].pt;

        if (fabs(pt0.y - pt1.y) <= row_align_pixel_threshold)
        {
            new_matches.push_back(matches[match_index]);
        }
    }

    matches.swap(new_matches);
}

void PointFeatureMatcher::geometricTest(const std::vector<cv::KeyPoint> &keypoints0,
                                        const std::vector<cv::KeyPoint> &keypoints1,
                                        std::vector<cv::DMatch> &matches,
                                        const float &ransac_reproj_th,
                                        const float &ransac_confidence,
                                        const float &sampson_error_sq_th)
{
    if (matches.size() < 8)
        return;

    // Build point arrays in match order
    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    points0.reserve(matches.size());
    points1.reserve(matches.size());

    for (const auto &m : matches)
    {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= (int)keypoints0.size() ||
            m.trainIdx >= (int)keypoints1.size())
        {
            // Skip invalid match indices (rare, but safe)
            continue;
        }
        points0.push_back(keypoints0[m.queryIdx].pt);
        points1.push_back(keypoints1[m.trainIdx].pt);
    }

    if (points0.size() < 8 || points1.size() < 8)
        return;

    // Fundamental matrix with RANSAC
    std::vector<uchar> mask;
    // cv::Mat F = cv::findFundamentalMat(points0, points1, cv::RANSAC, ransac_reproj_th, ransac_confidence, mask);
    cv::Mat F = cv::findFundamentalMat(points0, points1, cv::USAC_MAGSAC, ransac_reproj_th, ransac_confidence, mask);

    if (F.empty())
        return;

    if (mask.size() != points0.size())
        return;

    if (F.type() != CV_64F)
        F.convertTo(F, CV_64F);

    const cv::Mat Ft = F.t();

    // Output filtered matches
    std::vector<cv::DMatch> out;
    out.reserve(matches.size());

    // NOTE:
    // mask corresponds to points0/points1 which correspond to the *filtered valid matches above*
    // If you want mask to align with original matches 1:1, don't skip invalid indices earlier.
    // Here we choose safety; typically indices are valid so alignment is preserved.
    size_t idx = 0;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        const auto &m = matches[i];
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= (int)keypoints0.size() ||
            m.trainIdx >= (int)keypoints1.size())
        {
            continue;
        }

        if (!mask[idx])
        {
            ++idx;
            continue;
        }

        // RANSAC-only mode
        if (sampson_error_sq_th < 0.0f)
        {
            out.push_back(m);
            ++idx;
            continue;
        }

        // Do Sampson test
        const cv::Point2f p0 = keypoints0[m.queryIdx].pt;
        const cv::Point2f p1 = keypoints1[m.trainIdx].pt;

        const cv::Vec3d x1(p0.x, p0.y, 1.0);
        const cv::Vec3d x2(p1.x, p1.y, 1.0);

        // l2 = F * x1
        const cv::Vec3d l2(
            F.at<double>(0, 0) * x1[0] + F.at<double>(0, 1) * x1[1] + F.at<double>(0, 2),
            F.at<double>(1, 0) * x1[0] + F.at<double>(1, 1) * x1[1] + F.at<double>(1, 2),
            F.at<double>(2, 0) * x1[0] + F.at<double>(2, 1) * x1[1] + F.at<double>(2, 2));

        // l1 = F^T * x2
        const cv::Vec3d l1(
            Ft.at<double>(0, 0) * x2[0] + Ft.at<double>(0, 1) * x2[1] + Ft.at<double>(0, 2),
            Ft.at<double>(1, 0) * x2[0] + Ft.at<double>(1, 1) * x2[1] + Ft.at<double>(1, 2),
            Ft.at<double>(2, 0) * x2[0] + Ft.at<double>(2, 1) * x2[1] + Ft.at<double>(2, 2));

        const double e = x2.dot(l2); // x2^T F x1

        const double denom =
            l2[0] * l2[0] + l2[1] * l2[1] +
            l1[0] * l1[0] + l1[1] * l1[1];

        if (denom < 1e-12)
        {
            ++idx;
            continue;
        }

        const double sd = (e * e) / denom; // Sampson distance (approx squared error)

        if (sd < (double)sampson_error_sq_th)
            out.push_back(m);

        ++idx;
    }

    matches.swap(out);
}

void PointFeatureMatcher::geometricTest(const std::vector<cv::KeyPoint> &keypoints0,
                                        const std::vector<cv::KeyPoint> &keypoints1,
                                        std::vector<cv::DMatch> &matches,
                                        const Eigen::Matrix3d &K0,
                                        const Eigen::Matrix3d &K1,
                                        const float &ransac_reproj_th,
                                        const float &ransac_confidence,
                                        const float &sampson_error_sq_th)
{
    if (matches.size() < 5) // Essential matrix needs min 5 points
        return;

    // Convert Eigen K to cv::Mat
    cv::Mat cvK0 = (cv::Mat_<double>(3, 3) <<
        K0(0,0), K0(0,1), K0(0,2),
        K0(1,0), K0(1,1), K0(1,2),
        K0(2,0), K0(2,1), K0(2,2));

    cv::Mat cvK1 = (cv::Mat_<double>(3, 3) <<
        K1(0,0), K1(0,1), K1(0,2),
        K1(1,0), K1(1,1), K1(1,2),
        K1(2,0), K1(2,1), K1(2,2));

    // Build point arrays in match order
    std::vector<cv::Point2f> points0, points1;
    points0.reserve(matches.size());
    points1.reserve(matches.size());

    for (const auto &m : matches)
    {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= (int)keypoints0.size() ||
            m.trainIdx >= (int)keypoints1.size())
            continue;

        points0.push_back(keypoints0[m.queryIdx].pt);
        points1.push_back(keypoints1[m.trainIdx].pt);
    }

    if (points0.size() < 5)
        return;

    // pixel threshold → normalized units
    // normalized = pixel / focal_length
    // Use average focal length of both cameras as approximation
    const double fx0 = K0(0, 0);
    const double fy0 = K0(1, 1);
    const double fx1 = K1(0, 0);
    const double fy1 = K1(1, 1);

    const double meanFocal = (fx0 + fy0 + fx1 + fy1) / 4.0;
    const double normThresh = ransac_reproj_th / meanFocal;
    const double normSampsonTh = sampson_error_sq_th / (meanFocal * meanFocal);

    // Normalize to image coordinates using undistortPoints with P=NoArray()
    // This transforms points into normalized coords valid for identity K
    std::vector<cv::Point2f> normPts0, normPts1;
    cv::undistortPoints(points0, normPts0, cvK0, cv::noArray()); // no distortion, P=NoArray → normalized coords
    cv::undistortPoints(points1, normPts1, cvK1, cv::noArray());

    // Find Essential matrix using identity K (since points are already normalized)
    std::vector<uchar> mask;
    cv::Mat E = cv::findEssentialMat(normPts0, normPts1,
                                     cv::Mat::eye(3, 3, CV_64F), // identity K — points already normalized
                                     cv::USAC_MAGSAC,
                                     ransac_confidence,
                                     normThresh, // threshold is now in normalized image units
                                     mask);

    if (E.empty())
        return;

    if (mask.size() != points0.size())
        return;

    // E must be 3x3; handle the rare case of multiple E returned
    if (E.rows != 3 || E.cols != 3)
        E = E.rowRange(0, 3);

    if (E.type() != CV_64F)
        E.convertTo(E, CV_64F);

    // Sampson error uses the Essential matrix on normalized coordinates
    // E plays the same role as F but in normalized space
    const cv::Mat Et = E.t();

    std::vector<cv::DMatch> out;
    out.reserve(matches.size());

    size_t idx = 0;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        const auto &m = matches[i];
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= (int)keypoints0.size() ||
            m.trainIdx >= (int)keypoints1.size())
            continue;

        if (!mask[idx])
        {
            ++idx;
            continue;
        }

        // RANSAC-only mode
        if (sampson_error_sq_th < 0.0f)
        {
            out.push_back(m);
            ++idx;
            continue;
        }

        // Sampson error in normalized coordinates
        const cv::Point2f &np0 = normPts0[idx];
        const cv::Point2f &np1 = normPts1[idx];

        const cv::Vec3d x1(np0.x, np0.y, 1.0);
        const cv::Vec3d x2(np1.x, np1.y, 1.0);

        // l2 = E * x1
        const cv::Vec3d l2(
            E.at<double>(0,0)*x1[0] + E.at<double>(0,1)*x1[1] + E.at<double>(0,2),
            E.at<double>(1,0)*x1[0] + E.at<double>(1,1)*x1[1] + E.at<double>(1,2),
            E.at<double>(2,0)*x1[0] + E.at<double>(2,1)*x1[1] + E.at<double>(2,2));

        // l1 = E^T * x2
        const cv::Vec3d l1(
            Et.at<double>(0,0)*x2[0] + Et.at<double>(0,1)*x2[1] + Et.at<double>(0,2),
            Et.at<double>(1,0)*x2[0] + Et.at<double>(1,1)*x2[1] + Et.at<double>(1,2),
            Et.at<double>(2,0)*x2[0] + Et.at<double>(2,1)*x2[1] + Et.at<double>(2,2));

        const double e = x2.dot(l2); // x2^T E x1

        const double denom =
            l2[0]*l2[0] + l2[1]*l2[1] +
            l1[0]*l1[0] + l1[1]*l1[1];

        if (denom < 1e-12)
        {
            ++idx;
            continue;
        }

        const double sd = (e * e) / denom;

        if (sd < normSampsonTh)
            out.push_back(m);

        ++idx;
    }

    matches.swap(out);
}

void PointFeatureMatcher::robustMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<cv::DMatch>& good_matches, const float & lowe_ratio) const
{
    good_matches.clear();

    // 1. Match the two image descriptors
    std::vector<std::vector<cv::DMatch>> matches01, matches10;

    // 1a. From image 0 to image 1
    m_matcher->knnMatch(descriptors0, descriptors1, matches01, 2); // return 2 nearest neighbours

    // 1b. From image 2 to image 1
    m_matcher->knnMatch(descriptors1, descriptors0, matches10, 2); // return 2 nearest neighbours

    // 2. Remove matches for which NN ratio is > than threshold
    // clean image 0 -> image 1 matches
    ratio_test(matches01, lowe_ratio);
    // clean image 1 -> image 0 matches
    ratio_test(matches10, lowe_ratio);

    // 4. Remove non-symmetrical matches
    symmetry_test(matches01, matches10, good_matches);
}

void PointFeatureMatcher::robustMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<cv::DMatch>& good_matches, const float & lowe_ratio, const int & max_dist) const
{
    good_matches.clear();

    // 1. Match the two image descriptors
    std::vector<std::vector<cv::DMatch>> matches01, matches10;

    // 1a. From image 0 to image 1
    m_matcher->knnMatch(descriptors0, descriptors1, matches01, 2); // return 2 nearest neighbours

    // 1b. From image 2 to image 1
    m_matcher->knnMatch(descriptors1, descriptors0, matches10, 2); // return 2 nearest neighbours

    // 2. Remove matches for which NN ratio is > than threshold
    // clean image 0 -> image 1 matches
    ratio_test(matches01, lowe_ratio, max_dist);
    // clean image 1 -> image 0 matches
    ratio_test(matches10, lowe_ratio, max_dist);

    // 4. Remove non-symmetrical matches
    symmetry_test(matches01, matches10, good_matches);
}

void PointFeatureMatcher::knnMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<std::vector<cv::DMatch>> & matches01, const int & num_nearest_neighbors)
{
    matches01.clear();
    m_matcher->knnMatch(descriptors0, descriptors1, matches01, num_nearest_neighbors);
}

void PointFeatureMatcher::fastRobustMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<cv::DMatch>& good_matches, const float & lowe_ratio) const
{
    good_matches.clear();

    // 1. Match the two image descriptors
    std::vector<std::vector<cv::DMatch>> matches;
    m_matcher->knnMatch(descriptors0, descriptors1, matches, 2); // return 2 nearest neighbours

    // 2. Remove matches for which NN ratio is > than threshold
    ratio_test(matches, lowe_ratio);

    // 3. Fill good matches container
    for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
    {
        if (!matchIterator->empty())
        {
            good_matches.push_back((*matchIterator)[0]);
        }
    }
}

//-----------------------------------------------------------------
// PointFeatureMatcherBruteForce
//-----------------------------------------------------------------
PointFeatureMatcherBruteForce::PointFeatureMatcherBruteForce(int normType)
    : PointFeatureMatcher()
{
    m_matcher = cv::BFMatcher::create(normType, false); // cv::NORM_L2, cv::NORM_HAMMING
}

PointFeatureMatcherBruteForce::~PointFeatureMatcherBruteForce()
{

}

//-----------------------------------------------------------------
// PointFeatureMatcherFLANN
//-----------------------------------------------------------------
PointFeatureMatcherFLANN::PointFeatureMatcherFLANN()
    : PointFeatureMatcher()
{
    m_matcher = cv::FlannBasedMatcher::create();
}

PointFeatureMatcherFLANN::~PointFeatureMatcherFLANN()
{

}

//-----------------------------------------------------------------
// PointFeatureMatcherGMS
//-----------------------------------------------------------------
PointFeatureMatcherGMS::PointFeatureMatcherGMS(int normType, bool withRotation, bool withScale, double thresholdFactor)
    : PointFeatureMatcher()
    , m_withRotation(withRotation)
    , m_withScale(withScale)
    , m_thresholdFactor(thresholdFactor)
{
    // GMS filters an initial all-to-all BF match, so we always use a
    // cross-check-disabled L2 brute-force matcher underneath.
    m_matcher = cv::BFMatcher::create(normType, false); // cv::NORM_L2, cv::NORM_HAMMING
}

PointFeatureMatcherGMS::~PointFeatureMatcherGMS()
{
}

void PointFeatureMatcherGMS::matchGMS(const cv::Size &imageSize0,
                                      const cv::Size &imageSize1,
                                      const std::vector<cv::KeyPoint> &keypoints0,
                                      const std::vector<cv::KeyPoint> &keypoints1,
                                      const cv::Mat &descriptors0,
                                      const cv::Mat &descriptors1,
                                      std::vector<cv::DMatch> &good_matches,
                                      const int &max_dist) const
{
    good_matches.clear();

    // 1. Raw 1-NN brute-force match (GMS needs a dense, unfiltered set)
    std::vector<cv::DMatch> allMatches;
    m_matcher->match(descriptors0, descriptors1, allMatches);

    if (allMatches.empty())
        return;

    if (max_dist > 0)
    {
        allMatches.erase(
            std::remove_if(allMatches.begin(), allMatches.end(),
                           [&](const cv::DMatch &m) { return m.distance > max_dist; }),
            allMatches.end());
    }

    // 2. GMS filtering
    std::vector<cv::DMatch> gmsMatches;
    cv::xfeatures2d::matchGMS(imageSize0, imageSize1,
                              keypoints0, keypoints1,
                              allMatches, gmsMatches,
                              m_withRotation,
                              m_withScale,
                              m_thresholdFactor);

    good_matches.swap(gmsMatches);
}