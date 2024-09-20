#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // 添加此行
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <chrono>

using namespace std;
using namespace cv;

void buildPyramid(const Mat &src, vector<Mat> &pyramid, int levels, double scale) {
  pyramid.push_back(src);
  for (int i = 1; i < levels; ++i) {
    Mat dst;
    resize(pyramid[i - 1], dst, Size(), scale, scale);
    pyramid.push_back(dst);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- 初始化
  int pyramid_levels = 3; // 金字塔层数
  double scale = 0.5; // 缩放倍数
  vector<Mat> pyramid_1, pyramid_2;
  buildPyramid(img_1, pyramid_1, pyramid_levels, scale);
  buildPyramid(img_2, pyramid_2, pyramid_levels, scale);

  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  vector<DMatch> final_matches;

  for (int i = pyramid_levels - 1; i >= 0; --i) {
    //-- 第一步:检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(pyramid_1[i], keypoints_1);
    detector->detect(pyramid_2[i], keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(pyramid_1[i], keypoints_1, descriptors_1);
    descriptor->compute(pyramid_2[i], keypoints_2, descriptors_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
      if (matches[i].distance <= max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
      }
    }

    if (i == 0) {
      final_matches = good_matches;
    }
  }

  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  if (!final_matches.empty()) {
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, final_matches, img_goodmatch);
    if (!img_goodmatch.empty() && img_goodmatch.size().width > 0 && img_goodmatch.size().height > 0) {
      imshow("good matches", img_goodmatch);
    }
  }
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, final_matches, img_match);
  if (!img_match.empty() && img_match.size().width > 0 && img_match.size().height > 0) {
    imshow("all matches", img_match);
  }
  waitKey(0);

  //-- 计算位移
  if (!final_matches.empty()) {
    Point2f displacement(0, 0);
    for (const auto &match : final_matches) {
      displacement += keypoints_2[match.trainIdx].pt - keypoints_1[match.queryIdx].pt;
    }
    displacement /= static_cast<float>(final_matches.size()); // 修复类型不匹配
    cout << "Displacement: " << displacement << endl;
  }

  return 0;
}