import cv2
import numpy as np
import os
import time
from Image_Viewer import ImageViewer


class Frame:
    _id_counter = 0  # 类变量，用于计数
    def __init__(self, image_path):
        self.image_path = image_path  # 对应图像文件路径

        self.image = cv2.imread(image_path)  # 读取图像

        self.pyramid_images = self.generate_pyramid(self.image)  # 生成金字塔图像
        self.keypoints = []  # 特征点列表
        self.pose = None  # 位姿
        self.uv_pose = None  # 位姿在图像上的像素坐标，这里是在最高分辨率的图像上的像素坐标
        self.object_id = None  # 对象ID
        self.is_active = True  # 是否是活帧

        # 检测每层金字塔下的特征点
        for layer in range(len(self.pyramid_images)):
            self.detect_keypoints_at_layer(layer)
        
        self.object_id = Frame._id_counter
        Frame._id_counter += 1
        if self.object_id == 0:
            self.uv_pose = np.array([0, 0])

    # 生成图像金字塔,levels表示金字塔层数
    def generate_pyramid(self, image, levels=3):
        pyramid_images = [image]
        for _ in range(1, levels):
            # 使用 cv2.resize 函数将图像缩小 5 倍
            width = int(image.shape[1] / 3)
            height = int(image.shape[0] / 3)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            pyramid_images.append(image)
        return pyramid_images

    # 在指定层检测特征点
    def detect_keypoints_at_layer(self, layer):
        if self.pyramid_images is None or layer >= len(self.pyramid_images):
            raise ValueError("Invalid pyramid layer.")
        
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(self.pyramid_images[layer], None)
        
        for kp, desc in zip(keypoints, descriptors):
            self.keypoints.append(Keypoint(kp.pt, layer, self, desc))
        return self.keypoints

    def compute_pose(self):
        # 计算位姿的逻辑
        pass

    def update_uv_pose(self, uav_pose):
        # 更新位姿在图像上的像素坐标的逻辑
        self.uv_pose = uav_pose

class Keypoint:
    def __init__(self, pt, pyramid_layer, associated_image, descriptor, is_anomalous=False):
        self.pt = pt  # 特征点像素位置
        self.descriptor = descriptor  # 描述符
        self.pyramid_layer = pyramid_layer  # 特征点所在图像金字塔层
        self.associated_image = associated_image  # 关联图像
        self.is_anomalous = is_anomalous  # 是否为异常点

    def is_within_bounds(self, image_shape):
        x, y = self.pt
        return 0 <= x < image_shape[1] and 0 <= y < image_shape[0]

    def mark_anomalous(self):
        self.is_anomalous = True

class ActiveImage:
    def __init__(self, frames=[]):
        # if len(frames) < 5 or len(frames) > 7:  # 活动图像必须包含5到7个帧
        #     raise ValueError("ActiveImage must contain 5 to 7 frames.")
        self.frames = frames  # 5-7个连续的图像对象
        self.activate_len=7

    def add_frame(self, frame):
        if len(self.frames) >= self.activate_len:
            # 找到与当前帧 object_id 相差最大的帧
            max_diff_index = 0
            max_diff = abs(self.frames[0].object_id - frame.object_id)
            for i in range(1, len(self.frames)):
                diff = abs(self.frames[i].object_id - frame.object_id)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_index = i
            
            # 移除与当前帧 object_id 相差最大的帧
            self.frames[max_diff_index].is_active = False
            self.frames[max_diff_index].image = None  # 删除图像
            self.frames[max_diff_index].pyramid_images = None  # 删除图像
            self.frames.pop(max_diff_index)
        
        # 按照 object_id 顺序插入新帧
        inserted = False
        for i in range(len(self.frames)):
            if self.frames[i].object_id > frame.object_id:
                self.frames.insert(i, frame)
                inserted = True
                break
        if not inserted:
            self.frames.append(frame)

    def remove_frame(self, frame):
        frame.is_active=False
        frame.image = None  # 删除图像
        frame.pyramid_images = None  # 删除图像
        self.frames.remove(frame) #活动帧中移除
    def reactivate_frame(self, frame):
        frame.is_active=True
        frame.image = cv2.imread(frame.image_path)  # 读取图像
        frame.pyramid_images = frame.generate_pyramid(frame.image)  # 生成金字塔图像


    def get_active_frame_ids(self):
        return [frame.object_id for frame in self.frames]

class GlobalMap:
    def __init__(self, active_frames,all_frames,reduced_image=None, observation_frame_id=None,  uav_pose=None, stitched_active_frames=None):
        self.reduced_image = reduced_image  # 10倍缩小的图像拼接结果
        self.observation_frame_id = observation_frame_id  # 观察框所在图像ID
        self.active_frames = active_frames  # 活动图像列表
        self.uav_pose = uav_pose  # UAV位姿
        self.stitched_active_frames = stitched_active_frames  # 活动图像拼接结果
        self.all_frames=all_frames #所有帧对象

    def compute_translation(self):
        # 拼接活动图像的逻辑
        if not self.active_frames or len(self.active_frames.frames) == 0:
            raise ValueError("No active images to stitch.")
        if len(self.active_frames.frames)==1:
            return self.active_frames.frames[0].pyramid_images[2]
        # 获取所有活动帧的图像
        images = [frame.pyramid_images[2] for frame in self.active_frames.frames]
        keypoints_list = [frame.keypoints for frame in self.active_frames.frames]
    
        for i in range(1, len(images)):
            # 检查当前帧的uv_pose是否为None
            if self.active_frames.frames[i].uv_pose is not None:
                deta_pose = self.active_frames.frames[i].uv_pose-self.active_frames.frames[i-1].uv_pose
                # new_image = self.translation_and_warp(images[i-1], images[i], deta_pose)
                # print("直接拼接，不计算")
            else:
                # 获取当前图像和基准图像的特征点
                keypoints1 = [kp for kp in keypoints_list[i-1] if kp.pyramid_layer == 2]
                keypoints2 = [kp for kp in keypoints_list[i] if kp.pyramid_layer == 2]
                
                # 使用BFMatcher进行特征点匹配
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                descriptors1 = np.array([kp.descriptor for kp in keypoints1])
                descriptors2 = np.array([kp.descriptor for kp in keypoints2])
    
                matches = bf.match(descriptors1, descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # 获取匹配点
                points1 = [keypoints1[m.queryIdx].pt for m in matches]
                points2 = [keypoints2[m.trainIdx].pt for m in matches]
                
                # 计算平移并进行图像拼接
                deta_pose = self.compute_translation_and_warp(images[i-1], images[i], points1, points2)
                self.active_frames.frames[i].uv_pose = deta_pose + self.active_frames.frames[i-1].uv_pose
                print(f"帧{self.active_frames.frames[i].object_id}的uv_pose为{self.active_frames.frames[i].uv_pose}")

    # 计算特征点匹配的平移向量
    def compute_translation_and_warp(self, img_1, img_2, points1, points2):
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

class UAV_LOCATION:

    def __init__(self):
        self.frames = []
        self.active_frames = ActiveImage()  
        self.map=GlobalMap(self.active_frames,self.frames)
        self.map_img=None

    def read_frame(self, image_path):
        frame = Frame(image_path)
        self.frames.append(frame)
        self.active_frames.add_frame(frame)
        # 时刻取最新的地图，观察窗应该从最新帧id-活动地图帧数量开始
        self.map.observation_frame_id = max(0,self.frames[-1].object_id-self.active_frames.activate_len+1)
        self.map.compute_translation()

    
    def get_global_map(self):
        return self.map_img
    
    def get_map_observaion_id(self):
        return self.map.observation_frame_id
    
    def set_map_observaion_id(self,observation_frame_id):
        self.map.observation_frame_id=observation_frame_id



if __name__ == "__main__":
    image_dir = '/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/' 

    img_count = 0  
    uav=UAV_LOCATION()

    viewer = ImageViewer()

    
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".JPG"):
            img_count+=1
            if img_count>20:
                break
            path=os.path.join(image_dir, filename)
            uav.read_frame(path)
            viewer.add_image(path, uav.frames[-1].uv_pose*9)

    viewer.run()
    cv2.destroyAllWindows()


