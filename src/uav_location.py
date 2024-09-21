import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class Frame:
    def __init__(self, image, pyramid_images, keypoints, pose, object_id, is_active, image_path):
        self.image = image  # 图像
        self.pyramid_images = pyramid_images  # 不同金字塔层下的图像列表
        self.keypoints = keypoints  # 特征点列表
        self.pose = pose  # 位姿
        self.object_id = object_id  # 对象ID
        self.is_active = is_active  # 是否是活帧
        self.image_path = image_path  # 对应图像文件路径

class Keypoint:
    def __init__(self, pixel_position, pyramid_layer, associated_image, is_anomalous):
        self.pixel_position = pixel_position  # 特征点像素位置
        self.pyramid_layer = pyramid_layer  # 特征点所在图像金字塔层
        self.associated_image = associated_image  # 关联图像
        self.is_anomalous = is_anomalous  # 是否为异常点

class ActiveImage:
    def __init__(self, frames):
        if len(frames) < 5 or len(frames) > 7:  # 活动图像必须包含5到7个帧
            raise ValueError("ActiveImage must contain 5 to 7 frames.")
        self.frames = frames  # 5-7个连续的图像对象

class GlobalMap:
    def __init__(self, reduced_image, observation_frame_id, active_images, uav_pose, dead_frames, stitched_active_images):
        self.reduced_image = reduced_image  # 10倍缩小的图像拼接结果
        self.observation_frame_id = observation_frame_id  # 观察框所在图像ID
        self.active_images = active_images  # 活动图像列表
        self.uav_pose = uav_pose  # UAV位姿
        self.dead_frames = dead_frames  # 所有死帧对象
        self.stitched_active_images = stitched_active_images  # 活动图像拼接结果


# if __name__ == "__main__":
#     # 创建特征点对象
#     keypoint = Keypoint(pixel_position=(100, 150), pyramid_layer=2, associated_image="image_1.jpg", is_anomalous=False)
#     # 创建帧对象
#     frame = Frame(image="image_1.jpg", pyramid_images=["pyramid_1.jpg", "pyramid_2.jpg"], keypoints=[keypoint], pose="pose_1", object_id=1, is_active=True, image_path="/path/to/image_1.jpg")
#     # 创建活动图像对象
#     active_image = ActiveImage(frames=[frame, frame, frame, frame, frame])
#     # 创建全局地图对象
#     global_map = GlobalMap(reduced_image="reduced_image.jpg", observation_frame_id=1, active_images=[active_image], uav_pose="uav_pose_1", dead_frames=[frame], stitched_active_images="stitched_image.jpg")

img1 = cv2.imread("03_0001.JPG")
img2 = cv2.imread("03_0002.JPG")
img1=cv2.resize(img1, (img1.shape[1] // 8, img1.shape[0] // 8))
img2=cv2.resize(img2, (img2.shape[1] // 8, img2.shape[0] // 8))


K=[7744,0,3976,0,7744,2652,0,0,1]

R=[1,0,0,0,1,0,0,0,1]
t=[62.16487211,793.82361558,0]
# 计算H单应矩阵
# 相机内参矩阵
K = np.array([
    [7744, 0, 3976],
    [0, 7744, 2652],
    [0, 0, 1]
])

# 旋转矩阵
R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# 平移向量
t = np.array([62.16487211/10, 793.82361558/8, 0])

# 平面法向量
n = np.array([0, 0, 1])

# 平面到相机的距离
d = 1

# 计算H单应矩阵
H = K @ (R + np.outer(t, n) / d) @ np.linalg.inv(K)
H=H

# 创建窗口
cv2.namedWindow("Stitched Image")

# 滑动条的回调函数
def update(val):
    # 获取滑动条的值
    tx = cv2.getTrackbarPos('Tx', 'Stitched Image') - 1000
    ty = cv2.getTrackbarPos('Ty', 'Stitched Image') - 1000

    # 更新平移向量
    t = np.array([tx, ty, 0])

    # 构建仿射变换矩阵
    affine_matrix = np.hstack((R[:2, :2], t[:2].reshape(2, 1))).astype(np.float32)

    # 计算输出图像的尺寸
    output_shape = (1080, 1080)

    # 将第二张图像变换到第一张图像的坐标系中
    warped_img2 = cv2.warpAffine(img2, affine_matrix, output_shape)

    # 创建一个空白画布用于拼接图像
    stitched_image = np.zeros((1080, 1080, 3), dtype=np.uint8)

    # 将第一张图像放到画布的中心
    center_x = (1080 - img1.shape[1]) // 2
    center_y = (1080 - img1.shape[0]) // 2
    stitched_image[center_y:center_y + img1.shape[0], center_x:center_x + img1.shape[1]] = img1

    # 将变换后的第二张图像叠加到画布上
    x_offset = center_x + int(t[0])
    y_offset = center_y + int(t[1])
    for y in range(img2.shape[0]):
        for x in range(img2.shape[1]):
            if 0 <= y + y_offset < stitched_image.shape[0] and 0 <= x + x_offset < stitched_image.shape[1]:
                alpha = 0.5  # 透明度
                blended_pixel = cv2.addWeighted(
                    stitched_image[y + y_offset, x + x_offset].reshape(1, 1, 3), alpha,
                    warped_img2[y, x].reshape(1, 1, 3), 1 - alpha, 0)
                stitched_image[y + y_offset, x + x_offset] = blended_pixel.reshape(3)

    # 显示拼接后的图像
    cv2.imshow("Stitched Image", stitched_image)

# 创建滑动条
cv2.createTrackbar('Tx', 'Stitched Image', 0, 2000, update)
cv2.createTrackbar('Ty', 'Stitched Image', 0, 2000, update)

# 初始化显示
update(0)

# 等待用户退出
cv2.waitKey(0)
cv2.destroyAllWindows()