#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <filesystem>
#include <vector>
#include <opencv2/calib3d.hpp> // 用于计算单应矩阵

using namespace std;
using namespace cv;
namespace fs = std::filesystem; // 使用 C++17 的文件系统库

// 构建图像金字塔
void buildPyramid(const Mat &src, vector<Mat> &pyramid, int levels, double scale)
{
  pyramid.push_back(src);
  for (int i = 1; i < levels; ++i)
  {
    Mat dst;
    resize(pyramid[i - 1], dst, Size(), scale, scale); // 缩放图像
    pyramid.push_back(dst);
  }
}

// 计算图像之间的位移
Point2f computeDisplacement(const Mat &img_1, const Mat &img_2, int pyramid_levels, double scale, vector<DMatch> &final_matches, vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2)
{
  vector<Mat> pyramid_1, pyramid_2;
  buildPyramid(img_1, pyramid_1, pyramid_levels, scale);
  buildPyramid(img_2, pyramid_2, pyramid_levels, scale);

  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  for (int i = pyramid_levels - 1; i >= 0; --i)
  {
    detector->detect(pyramid_1[i], keypoints_1); // 检测特征点
    detector->detect(pyramid_2[i], keypoints_2);

    descriptor->compute(pyramid_1[i], keypoints_1, descriptors_1); // 计算描述子
    descriptor->compute(pyramid_2[i], keypoints_2, descriptors_2);

    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches); // 匹配特征点

    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2)
                                  { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
      if (matches[i].distance <= max(2 * min_dist, 30.0))
      {
        good_matches.push_back(matches[i]); // 筛选好的匹配点
      }
    }

    if (i == 0)
    {
      final_matches = good_matches; // 保存最终的匹配点
    }
  }

  Point2f displacement(0, 0);
  if (!final_matches.empty())
  {
    for (const auto &match : final_matches)
    {
      displacement += keypoints_2[match.trainIdx].pt - keypoints_1[match.queryIdx].pt; // 计算位移
    }
    displacement /= static_cast<float>(final_matches.size()); // 取平均位移
  }

  return displacement;
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cout << "usage: feature_extraction img_directory" << endl;
    return 1;
  }

  string img_directory = argv[1];
  vector<string> img_files;
  for (const auto &entry : fs::directory_iterator(img_directory))
  {
    img_files.push_back(entry.path().string()); // 获取目录中的所有图片文件
  }

  if (img_files.size() < 2)
  {
    cout << "Not enough images in the directory." << endl;
    return 1;
  }

  int pyramid_levels = 5;     // 金字塔层数
  double scale = 0.5;         // 缩放倍数
  double scale_factor = 0.01; // 缩放因子，用于调整位移的大小
  vector<Point2f> displacements;
  Point2f current_position(0, 0);

  namedWindow("Trajectory", WINDOW_AUTOSIZE); // 创建显示轨迹的窗口
  namedWindow("Image", WINDOW_AUTOSIZE);      // 创建显示图片的窗口
  Mat traj = Mat::zeros(600, 600, CV_8UC3);   // 初始化轨迹图像

  for (size_t i = 1; i < img_files.size(); ++i)
  {
    Mat img_1 = imread(img_files[i - 1], IMREAD_COLOR);
    Mat img_2 = imread(img_files[i], IMREAD_COLOR);
    if (img_1.empty() || img_2.empty())
    {
      cout << "Could not open or find the image!" << endl;
      return -1;
    }

    // 定义全局地图
    int map_width = 1500, map_height = 1500;                      // 调整地图大小
    Mat global_map = Mat::zeros(map_height, map_width, CV_8UC3);  // 根据需要调整大小
    Point2f map_offset(global_map.cols / 2, global_map.rows / 2); // 地图偏移量

    vector<DMatch> final_matches;
    vector<KeyPoint> keypoints_1, keypoints_2;
    Point2f displacement = computeDisplacement(img_1, img_2, pyramid_levels, scale, final_matches, keypoints_1, keypoints_2);
    current_position += displacement; // 更新当前位置
    displacements.push_back(current_position);

    // 绘制轨迹
    Point2f draw_position = current_position * scale_factor + Point2f(traj.cols / 2, traj.rows / 2);
    cout << "Displacement: " << displacement << ", Current position: " << current_position << endl;
    circle(traj, draw_position, 1, Scalar(0, 0, 255), 2); // 在轨迹图像上绘制当前位置
    imshow("Trajectory", traj);                           // 显示轨迹图像

    // 将 KeyPoint 转换为 Point2f
    vector<Point2f> points1, points2;
    for (const auto &kp : keypoints_1)
    {
      points1.push_back(kp.pt);
    }
    for (const auto &kp : keypoints_2)
    {
      points2.push_back(kp.pt);
    }

    // 将对齐好的图片叠放在全局地图上
    Mat img_transformed;
    Mat H = findHomography(points1, points2, cv::RANSAC); // 计算单应矩阵
    warpPerspective(img_2, img_transformed, H, global_map.size(), INTER_LINEAR, BORDER_TRANSPARENT);

    // 叠加图像时，不使用透明度融合，直接将变换后的图像插入全局地图
    img_transformed.copyTo(global_map, img_transformed); // 使用掩膜确保透明区域不影响原始地图

    // 在全局地图上显示当前位置
    Point2f map_position = current_position * scale_factor + map_offset;
    circle(global_map, map_position, 5, Scalar(0, 0, 255), -1); // 在全局地图上绘制当前位置
    imshow("Global Map", global_map);                           // 显示全局地图

    // 显示匹配特征点的图片
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, final_matches, img_matches); // 绘制匹配特征点
    Mat img_small;
    resize(img_matches, img_small, Size(), 0.5, 0.5); // 缩小图片
    imshow("Image", img_small);                       // 显示缩小后的图片

    waitKey(0); // 等待100毫秒
  }

  waitKey(0); // 等待用户按键
  return 0;
}