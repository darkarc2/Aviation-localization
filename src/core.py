import cv2
import numpy as np
import os
import time
from show_img import ImageViewer






def compute_translation(points1,points2):
	
    # 只使用前20个匹配点
    points1 = np.float32(points1[:20]).reshape(-1, 2)
    points2 = np.float32(points2[:20]).reshape(-1, 2)

    # 计算匹配点连线的方向
    directions = []
    for pt1, pt2 in zip(points1, points2):
        direction = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        directions.append(direction)

    # 统计方向分布
    directions = np.array(directions)
    num_bins = 36  # 将方向分成36个区间，每个区间10度
    hist, bin_edges = np.histogram(directions, bins=num_bins)

    # 找出主要方向
    main_direction_idx = np.argmax(hist)
    main_direction = (bin_edges[main_direction_idx] + bin_edges[main_direction_idx + 1]) / 2

    # 删除偏差较大的点
    threshold = np.pi / 18  # 10度的偏差
    filtered_points1 = []
    filtered_points2 = []
    for pt1, pt2, direction in zip(points1, points2, directions):
        if abs(direction - main_direction) <= threshold:
            filtered_points1.append(pt1)
            filtered_points2.append(pt2)

    filtered_points1 = np.array(filtered_points1)
    filtered_points2 = np.array(filtered_points2)

    # 重新计算平移向量
    translation_vector = -1 * np.mean(filtered_points2 - filtered_points1, axis=0)

    # 保存平移向量到pose变量
    deta_pose = translation_vector
    return deta_pose





# 读取图像
image_path1= "/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG"
img1 = cv2.imread(image_path1,0)
# 读取图像
image_path2 = "/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG"
img2 = cv2.imread(image_path2,0)


# 创建ORB检测器
orb = cv2.ORB_create()

# 检测关键点和描述符
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)



# 初始化
#沿着极线跟踪点
#等点深度稳定，认为成熟点，



import cv2
import numpy as np
import sophuspy as sp
from scipy.optimize import least_squares

# 相机内参
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
baseline = 0.573

# 双线性插值获取像素值
def get_pixel_value(img, x, y):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= img.shape[1]:
        x = img.shape[1] - 1
    if y >= img.shape[0]:
        y = img.shape[0] - 1
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, img.shape[1] - 1)
    y1 = min(y0 + 1, img.shape[0] - 1)
    a = x - x0
    b = y - y0
    return (1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x1] + (1 - a) * b * img[y1, x0] + a * b * img[y1, x1]

# 计算雅可比矩阵和误差
def compute_jacobian_and_error(params, img1, img2, px_ref, depth_ref):
    T21 = sp.SE3.exp(params)
    H = np.zeros((6, 6))
    b = np.zeros(6)
    cost = 0
    for i, (px, depth) in enumerate(zip(px_ref, depth_ref)):
        point_ref = np.array([(px[0] - cx) / fx, (px[1] - cy) / fy, 1]) * depth
        point_cur = T21 * point_ref
        Z = point_cur[2]
        X = point_cur[0]
        Y = point_cur[1]
        if Z < 0:
            continue
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        if u < 0 or u >= img2.shape[1] or v < 0 or v >= img2.shape[0]:
            continue
        error = get_pixel_value(img1, px[0], px[1]) - get_pixel_value(img2, u, v)
        J_img_pixel = np.array([
            0.5 * (get_pixel_value(img2, u + 1, v) - get_pixel_value(img2, u - 1, v)),
            0.5 * (get_pixel_value(img2, u, v + 1) - get_pixel_value(img2, u, v - 1))
        ])
        J_pixel_xi = np.array([
            [fx / Z, 0, -fx * X / (Z ** 2), -fx * X * Y / (Z ** 2), fx + fx * (X ** 2) / (Z ** 2), -fx * Y / Z],
            [0, fy / Z, -fy * Y / (Z ** 2), -fy - fy * (Y ** 2) / (Z ** 2), fy * X * Y / (Z ** 2), fy * X / Z]
        ])
        J = -J_img_pixel @ J_pixel_xi
        H += J.T @ J
        b += -error * J
        cost += error ** 2
    return H, b, cost

# 直接法位姿估计
def direct_pose_estimation_single_layer(img1, img2, px_ref, depth_ref, T21):
    iterations = 10
    last_cost = float('inf')
    for iter in range(iterations):
        H, b, cost = compute_jacobian_and_error(T21.log(), img1, img2, px_ref, depth_ref)
        if np.isnan(H).any() or np.isnan(b).any():
            print("Update is nan")
            break
        
        # 添加正则化项
        H += np.eye(6) * 1e-6
        
        try:
            update = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            print("Singular matrix, using least squares solution")
            update, _, _, _ = np.linalg.lstsq(H, b, rcond=None)
        
        T21 = sp.SE3.exp(update) * T21
        if cost > last_cost:
            print(f"Cost increased: {cost}, {last_cost}")
            break
        if np.linalg.norm(update) < 1e-3:
            print("Converged")
            break
        last_cost = cost
        print(f"Iteration: {iter}, cost: {cost}")
    print("T21 = \n", T21.matrix())
    return T21

# 主函数
def main():
    img1 = cv2.imread('./imgs/left.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./imgs/000002.png', cv2.IMREAD_GRAYSCALE)
    disparity = cv2.imread('./imgs/disparity.png', cv2.IMREAD_GRAYSCALE)
    px_ref = []
    depth_ref = []
    for y in range(20, img1.shape[0] - 20,50):
        for x in range(20, img1.shape[1] - 20,50):
            d = disparity[y, x]
            if d == 0:
                continue
            depth = fx * baseline / d
            px_ref.append([y, x])
            depth_ref.append(depth)
    print("Number of points: ", len(px_ref))
    T21 = sp.SE3()
    T21 = direct_pose_estimation_single_layer(img1, img2, px_ref, depth_ref, T21)
    print("Estimated pose:\n", T21.matrix())

if __name__ == '__main__':
    main()