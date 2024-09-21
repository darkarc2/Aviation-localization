import cv2
import numpy as np
import os

def load_images(image_dir):
    img_count = 0  
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".JPG"):
            img = cv2.imread(os.path.join(image_dir, filename))
            if img is not None:
                img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
                img_count += 1
                if img_count > 10:
                    break
                yield img

class ImageStitcher:
    def __init__(self):
        self.result = None
        self.trajectory = [(0, 0)]

    # 提取关键点和匹配
    def find_keypoints_and_matches(self, img1, img2):
        # 初始化ORB检测器
        orb = cv2.ORB_create()

        # 检测关键点和计算描述符
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # 使用BFMatcher进行匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # 按照距离排序
        matches = sorted(matches, key=lambda x: x.distance)

        return keypoints1, keypoints2, matches

    # 计算单应性矩阵并进行图像拼接
    def compute_homography_and_warp(self, img_1, img_2, points1, points2):
        # 将匹配点转换为 NumPy 数组
        points1 = np.float32(points1).reshape(-1, 1, 2)
        points2 = np.float32(points2).reshape(-1, 1, 2)
    
        # 使用RANSAC计算单应性矩阵
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    
        # 获取原始图像的高宽
        h1, w1 = img_1.shape[:2]
        h2, w2 = img_2.shape[:2]
    
        # 获取两幅图的边界坐标,reshape(-1, 1, 2)表示将坐标转换为三维数组
        img1_pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
        img2_pts = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
    
        # 获取 img1 的边界坐标变换之后的坐标
        img1_transform = cv2.perspectiveTransform(img1_pts, H)
    
        # 把 img2 和转换后的边界坐标连接起来
        result_pts = np.concatenate((img2_pts, img1_transform), axis=0)
    
        # 获取拼接图像的边界
        [x_min, y_min] = np.int32(result_pts.min(axis=0).ravel() - 1)
        [x_max, y_max] = np.int32(result_pts.max(axis=0).ravel() + 1)
        print(x_min, y_min, x_max, y_max)
        if x_min != -1 or y_min != -1:
            return None
    
        # 手动构造平移矩阵
        M = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
        # 对 img1 进行平移和透视操作
        result = cv2.warpPerspective(img_1, M.dot(H), (x_max - x_min, y_max - y_min))
    
        # 将 img2 叠加到 result 上
        result[-y_min:h2-y_min, -x_min:w2-x_min] = img_2
    
        return result

    # 计算位移
    def calculate_displacement(self, kp1, kp2, matches):
        directions = []
        for match in matches:
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt)
            direction = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            directions.append(direction)

        directions = np.array(directions)
        num_bins = 36  # 将方向分成36个区间，每个区间10度
        hist, bin_edges = np.histogram(directions, bins=num_bins)

        main_direction_idx = np.argmax(hist)
        main_direction = (bin_edges[main_direction_idx] + bin_edges[main_direction_idx + 1]) / 2

        threshold = np.pi / 18  # 10度的偏差
        filtered_matches = []
        for match, direction in zip(matches, directions):
            if abs(direction - main_direction) <= threshold:
                filtered_matches.append(match)

        displacement = np.zeros(2)
        if filtered_matches:
            for match in filtered_matches:
                displacement += np.array(kp2[match.trainIdx].pt) - np.array(kp1[match.queryIdx].pt)
            displacement /= len(filtered_matches)

        return displacement
    
    # 拼接图像
    def main(self, images):
        try:
            previous_image = next(images)
        except StopIteration:
            print("No images found in the directory.")
            return None

        self.result = previous_image
        for i, current_image in enumerate(images, start=1):
            kp1, kp2, matches = self.find_keypoints_and_matches(previous_image, current_image)
            if len(matches) < 4:
                print(f"Insufficient matches between images {i-1} and {i}. Skipping this image.")
                continue
            displacement = self.calculate_displacement(kp1, kp2, matches)
            self.trajectory.append(self.trajectory[-1] + displacement)
            
            # 提取匹配点的坐标
            points1 = [kp1[match.queryIdx].pt for match in matches]
            points2 = [kp2[match.trainIdx].pt for match in matches]
            
            self.result = self.compute_homography_and_warp(previous_image, current_image, points1, points2)
            if self.result is None:
                print(f"Failed to stitch images {i-1} and {i}. Restarting stitching from image {i}.")
                # 保留之前的全局地图，并在空一点的位置重新开始
                offset = np.array([100, 100])  # 偏移量，可以根据需要调整
                self.trajectory.append(self.trajectory[-1] + offset)
                previous_image = current_image
                self.result = cv2.copyMakeBorder(self.result, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                cv2.imshow("Stitched Image", self.result)
                self.draw_trajectory()
                cv2.waitKey(0)
                continue
            print(f"Stitched images {i-1} and {i}., pos: {self.trajectory[-1]}")
            # 更新 previous_image 为当前图像
            previous_image = self.result
            # result_show = cv2.resize(self.result, (self.result.shape[1] // 8, self.result.shape[0] // 8))
            cv2.imshow("Stitched Image", self.result)
            self.draw_trajectory()
            cv2.waitKey(0)
        return self.result
    
    # 绘制轨迹
    def draw_trajectory(self):
        traj_image = np.zeros((600, 600, 3), dtype=np.uint8)
        for i in range(1, len(self.trajectory)):
            start_point = (int(self.trajectory[i-1][0] / 8), int(self.trajectory[i-1][1] / 8))
            end_point = (int(self.trajectory[i][0] / 8), int(self.trajectory[i][1] / 8))
            cv2.line(traj_image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Trajectory", traj_image)

# 示例调用
image_dir = '/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/'
images = load_images(image_dir)
stitcher = ImageStitcher()
result = stitcher.main(images)

if result is not None:
    cv2.destroyAllWindows()