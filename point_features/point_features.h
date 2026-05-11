#ifndef _POINT_FEATURES_H
#define _POINT_FEATURES_H

#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Dense>

//-----------------------------------------------------------------
// PointFeatureExtractor (Abstract Class)
//-----------------------------------------------------------------
class PointFeatureExtractor
{
public:
    PointFeatureExtractor();
    virtual ~PointFeatureExtractor() = 0;

    virtual bool extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor = nullptr) = 0;
    virtual bool computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor) = 0;

    static void drawKeypoints(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & outImage);

protected:
    cv::Ptr<cv::FeatureDetector> m_detector;
    cv::Ptr<cv::DescriptorExtractor> m_extractor;
};

//-----------------------------------------------------------------
// PointFeatureExtractorSift
//-----------------------------------------------------------------
class PointFeatureExtractorSift : public PointFeatureExtractor
{
public:

    struct Parameters
    {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.03;
        double edgeThreshold = 10;
        double sigma = 1.6;
        int descriptorType = CV_32F; // CV_32F, CV_8U

        Parameters(const int nfeatures_ = 0, const int nOctaveLayers_ = 3, const double contrastThreshold_ = 0.05, const double edgeThreshold_ = 10.0, const double sigma_ = 1.6, const int descriptorType_ = CV_32F)
        {
            nfeatures = nfeatures_;
            nOctaveLayers = nOctaveLayers_;
            contrastThreshold = contrastThreshold_;
            edgeThreshold = edgeThreshold_;
            sigma = sigma_;
            descriptorType = descriptorType_; // CV_32F, CV_8U
        }
    };

    PointFeatureExtractorSift(const Parameters & params = Parameters());
    virtual ~PointFeatureExtractorSift();

    void setParams(const Parameters & params);
    bool extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor = nullptr);
    bool computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor);

};

//-----------------------------------------------------------------
// PointFeatureExtractorSurf
//-----------------------------------------------------------------
class PointFeatureExtractorSurf : public PointFeatureExtractor
{
public:

    struct Parameters
    {
        double hessianThreshold = 100;
        int nOctaves = 4;
        int nOctaveLayers = 3;
        bool extended = false;
        bool upright = false;

        Parameters(double hessianThreshold_ = 100, const int nOctaves_ = 4, const int nOctaveLayers_ = 3, const bool extended_ = false, const bool upright_ = false)
        {
            hessianThreshold = hessianThreshold_;
            nOctaves = nOctaves_;
            nOctaveLayers = nOctaveLayers_;
            extended = extended_;
            upright = upright_;
        }
    };

    PointFeatureExtractorSurf(const Parameters & params = Parameters());
    virtual ~PointFeatureExtractorSurf();

    void setParams(const Parameters & params);
    bool extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor = nullptr);
    bool computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor);

};

//-----------------------------------------------------------------
// PointFeatureExtractorORB
//-----------------------------------------------------------------
class PointFeatureExtractorORB : public PointFeatureExtractor
{
public:

    struct Parameters
    {
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;
        float keypoint_min_size_pixels = 0.0f;

        Parameters(const int nfeatures_ = 500,
                   const float scaleFactor_ = 1.2f,
                   const int nlevels_ = 8,
                   const int edgeThreshold_ = 31,
                   const int firstLevel_ = 0,
                   const int WTA_K_ = 2,
                   const int scoreType_ = cv::ORB::HARRIS_SCORE,
                   const int patchSize_ = 31,
                   const int fastThreshold_ = 20,
                   const float keypoint_min_size_pixels_ = 0.0f)
        {
            nfeatures = nfeatures_;
            scaleFactor = scaleFactor_;
            nlevels = nlevels_;
            edgeThreshold = edgeThreshold_;
            firstLevel = firstLevel_;
            WTA_K = WTA_K_;
            scoreType = scoreType_;
            patchSize = patchSize_;
            fastThreshold = fastThreshold_;
            keypoint_min_size_pixels = keypoint_min_size_pixels_;
        }
    };

    PointFeatureExtractorORB(const Parameters & _params = Parameters());
    virtual ~PointFeatureExtractorORB();

    void setParams(const Parameters & _params);
    bool extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor = nullptr);
    bool computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor);

private:
    Parameters params;

};

//-----------------------------------------------------------------
// GridingFeatureExtractorORB
//-----------------------------------------------------------------
class GridingFeatureExtractorORB : public PointFeatureExtractor
{
public:

    struct Parameters
    {
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;
        float keypoint_min_size_pixels = 0;

        Parameters(const int nfeatures_ = 500,
                   const float scaleFactor_ = 1.2f,
                   const int nlevels_ = 8,
                   const int edgeThreshold_ = 31,
                   const int firstLevel_ = 0,
                   const int WTA_K_ = 2,
                   const int scoreType_ = cv::ORB::HARRIS_SCORE,
                   const int patchSize_ = 31,
                   const int fastThreshold_ = 20,
                   const float keypoint_min_size_pixels_ = 0)
        {
            nfeatures = nfeatures_;
            scaleFactor = scaleFactor_;
            nlevels = nlevels_;
            edgeThreshold = edgeThreshold_;
            firstLevel = firstLevel_;
            WTA_K = WTA_K_;
            scoreType = scoreType_;
            patchSize = patchSize_;
            fastThreshold = fastThreshold_;
            keypoint_min_size_pixels = keypoint_min_size_pixels_;
        }
    };

    GridingFeatureExtractorORB(const int grid_x_ = 10, const int grid_y_ = 7, const Parameters & params = Parameters());
    virtual ~GridingFeatureExtractorORB();

    void setParams(const Parameters & params);
    bool extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor = nullptr);
    bool computeDescriptor(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat & descriptor);

private:
    Parameters params;
    int num_features_grid;
    int grid_x;
    int grid_y;
};

//-----------------------------------------------------------------
// GridingFeatureExtractor
//-----------------------------------------------------------------
class GridingFeatureExtractorFASTBrief
{
public:
    GridingFeatureExtractorFASTBrief(const int num_features_ = 100, const int grid_x_ = 10, const int grid_y_ = 7, const int threshold_ = 10, const bool nonmaxSuppression_ = true, const bool doCornerSubpix_ = true);
    virtual ~GridingFeatureExtractorFASTBrief();

    bool extract(const cv::Mat & image, std::vector<cv::KeyPoint> &keyPoints, cv::Mat * descriptor = nullptr, const cv::Mat & mask = cv::Mat());

private:
    int num_features;
    int grid_x;
    int grid_y;
    int threshold;
    bool nonmaxSuppression;
    bool doCornerSubpix;
    cv::Ptr<cv::DescriptorExtractor> feat_extractor;
};

//-----------------------------------------------------------------
// PointFeatureMatcher (Abstract Class)
//-----------------------------------------------------------------
class PointFeatureMatcher
{
public:
    PointFeatureMatcher();
    virtual ~PointFeatureMatcher() = 0;

    // Match feature points using ratio and symmetry test
    void robustMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<cv::DMatch>& good_matches, const float & lowe_ratio) const;
    void robustMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<cv::DMatch>& good_matches, const float & lowe_ratio, const int & max_dist) const;

    // Match feature points using ratio test
    void fastRobustMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<cv::DMatch>& good_matches, const float & lowe_ratio) const;

    // k-nearest neighbor matching
    void knnMatch(const cv::Mat & descriptors0, const cv::Mat & descriptors1, std::vector<std::vector<cv::DMatch>> & matches01, const int & num_nearest_neighbors = 2);

    // Draw matches
    static void drawMatches(const cv::Mat & image0, const std::vector<cv::KeyPoint> &keyPoints0,
                            const cv::Mat & image1, const std::vector<cv::KeyPoint> &keyPoints1,
                            const std::vector<cv::DMatch>& matches,
                            cv::Mat & outImage, const int thickness = 1);

    // Convert matches to correspondence
    static void matchesToPointCorrespondences(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                              const std::vector<cv::DMatch> &matches, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> &pt_correspondences,
                                              const double & point_scale_factor = 1.0);

    static void matchesToPointCorrespondences(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                              const std::vector<cv::DMatch> &matches, std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> &pt_correspondences,
                                              const float & point_scale_factor = 1.0f);

    static void matchesToPoints(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                                const std::vector<cv::DMatch> &matches,
                                std::vector<cv::Point2f> &points1,
                                std::vector<cv::Point2f> &points2,
                                const float &point_scale_factor = 1.0);

    // Row align test
    static void rowAlignTest(const std::vector<cv::KeyPoint> &keypoints0, const std::vector<cv::KeyPoint> &keypoints1,
                             std::vector<cv::DMatch> &matches, const float row_align_pixel_threshold = 5.0f);

    static void geometricTest(const std::vector<cv::KeyPoint> &keypoints0,
                              const std::vector<cv::KeyPoint> &keypoints1,
                              std::vector<cv::DMatch> &matches,
                              const float &ransac_reproj_th = 3.0,
                              const float &ransac_confidence = 0.99,
                              const float &sampson_error_sq_th = 4.0);

    static void geometricTest(const std::vector<cv::KeyPoint> &keypoints0,
                              const std::vector<cv::KeyPoint> &keypoints1,
                              std::vector<cv::DMatch> &matches,
                              const Eigen::Matrix3d &K0,
                              const Eigen::Matrix3d &K1,
                              const float &ransac_reproj_th = 3.0,
                              const float &ransac_confidence = 0.99,
                              const float &sampson_error_sq_th = 4.0);

  protected:
    cv::Ptr<cv::DescriptorMatcher> m_matcher;

private:
    // Clear matches for which NN ratio is > than threshold
    // return the number of removed points
    // (corresponding entries being cleared,
    // i.e. size will be 0)
    static int ratio_test(std::vector<std::vector<cv::DMatch> > & matches, const float & lowe_ratio);
    static int ratio_test(std::vector<std::vector<cv::DMatch>> &matches, const float &lowe_ratio, const int &max_dist);

    // Symmetrical test
    static void symmetry_test(const std::vector<std::vector<cv::DMatch>> &matches01,
                              const std::vector<std::vector<cv::DMatch>> &matches10,
                              std::vector<cv::DMatch> &symMatches01);
};

//-----------------------------------------------------------------
// PointFeatureMatcherBruteForce
//-----------------------------------------------------------------
class PointFeatureMatcherBruteForce : public PointFeatureMatcher
{
public:
    PointFeatureMatcherBruteForce(int normType = cv::NORM_L2); // if ORB, use cv::NORM_HAMMING
    ~PointFeatureMatcherBruteForce();
};

//-----------------------------------------------------------------
// PointFeatureMatcherFLANN
//-----------------------------------------------------------------
class PointFeatureMatcherFLANN : public PointFeatureMatcher
{
public:
    PointFeatureMatcherFLANN();
    ~PointFeatureMatcherFLANN();
};

//-----------------------------------------------------------------
// PointFeatureMatcherGMS (GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence)
//-----------------------------------------------------------------
class PointFeatureMatcherGMS : public PointFeatureMatcher
{
  public:
    // withRotation / withScale: enable if camera motion may include those
    PointFeatureMatcherGMS(int normType = cv::NORM_L2, // if ORB, use cv::NORM_HAMMING
                           bool withRotation = false,
                           bool withScale = false,
                           double thresholdFactor = 6.0);
    ~PointFeatureMatcherGMS();

    // GMS requires image sizes and keypoints to compute the grid statistics.
    // Falls back to a BF ratio-test match if GMS yields too few inliers.
    void matchGMS(const cv::Size &imageSize0,
                  const cv::Size &imageSize1,
                  const std::vector<cv::KeyPoint> &keypoints0,
                  const std::vector<cv::KeyPoint> &keypoints1,
                  const cv::Mat &descriptors0,
                  const cv::Mat &descriptors1,
                  std::vector<cv::DMatch> &good_matches,
                  const int &max_dist = -1) const;

  private:
    bool m_withRotation;
    bool m_withScale;
    double m_thresholdFactor;
};

#endif