import numpy as np
from modules.xfeat import XFeat
import tools
import cv2
from show_img import ImageViewer

xfeat = XFeat()

def gauss_newton_rotation(matched_keypoints1, matched_keypoints2, max_iterations=100, tol=1e-6):
    # 初始化旋转矩阵为单位矩阵
    R = np.eye(2)
    
    # 计算质心
    centroid1 = np.mean(matched_keypoints1, axis=0)
    centroid2 = np.mean(matched_keypoints2, axis=0)

    # 去中心化
    centered_keypoints1 = matched_keypoints1 - centroid1
    centered_keypoints2 = matched_keypoints2 - centroid2

    for iteration in range(max_iterations):
        # 计算当前旋转矩阵下的误差
        rotated_keypoints1 = centered_keypoints1 @ R.T
        error = centered_keypoints2 - rotated_keypoints1

        # 计算误差的范数
        error_norm = np.linalg.norm(error)
        if error_norm < tol:
            break

        # 计算雅可比矩阵
        J = np.zeros((2 * len(matched_keypoints1), 2))
        for i in range(len(matched_keypoints1)):
            J[2*i:2*i+2, :] = np.array([[-centered_keypoints1[i, 1], centered_keypoints1[i, 0]],
                                        [centered_keypoints1[i, 0], centered_keypoints1[i, 1]]])

        # 计算更新量
        delta = np.linalg.lstsq(J, error.flatten(), rcond=None)[0]

        # 更新旋转矩阵
        theta = np.arctan2(delta[1], delta[0])
        R_update = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        R = R @ R_update
    theat=np.arctan2(R[1, 0], R[0, 0])
    


    # H, mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)

    # # # 使用 mask 筛选出匹配的特征点
    # matched_keypoints1 = [points1[i] for i in range(len(mask)) if mask[i]]
    # matched_keypoints2 = [points2[i] for i in range(len(mask)) if mask[i]]
    # matched_keypoints1 = np.array(matched_keypoints1)
    # matched_keypoints2 = np.array(matched_keypoints2)
    t = np.mean(matched_keypoints2-matched_keypoints1,axis=0)
    return theat, -1*t


viewer = ImageViewer()



img1_path='/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0240.JPG'
img2_path='/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0241.JPG'

# 读取图像
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 匹配特征点
# points1, points2 = xfeat.match_lighterglue(img1, img2)
# keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in points1]
# keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in points2]
# Inference with batch = 1
# import time
# start = time.time()
# output0 = xfeat.detectAndCompute(img1, top_k = 1024)[0]
# output1 = xfeat.detectAndCompute(img2, top_k = 1024)[0]

# #Update with image resolution (required)
# output0.update({'image_size': (img1.shape[1], img1.shape[0])})
# output1.update({'image_size': (img2.shape[1], img2.shape[0])})
# points1, points2 = xfeat.match_lighterglue(output0, output1)
# end = time.time()
# print("time:",end-start)
# # orb匹配特征点
orb = cv2.ORB_create()
points1, descriptors1 = orb.detectAndCompute(img1, None)
points2, descriptors2 = orb.detectAndCompute(img2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
points1 = np.array([points1[m.queryIdx].pt for m in matches[:20]])
points2 = np.array([points2[m.trainIdx].pt for m in matches[:20]])

# 计算旋转矩阵和质心
theta, t = gauss_newton_rotation(points1, points2)

# 拼接图像

viewer.add_image(img1_path, np.array([0, 0,0]))
viewer.add_image(img2_path, np.array([t[0],t[1],theta]))


viewer.run()