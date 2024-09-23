import cv2
import numpy as np
import os
import time
from Image_Viewer import ImageViewer
from Frame_class import Frame, Keypoint, ActiveImage
import threading

class GlobalMap:
    def __init__(self, active_frames,all_frames,reduced_image=None, observation_frame_id=None,  uav_pose=None, stitched_active_frames=None):
        self.reduced_image = reduced_image  # 10倍缩小的图像拼接结果
        self.observation_frame_id = observation_frame_id  # 观察框所在图像ID
        self.active_frames = active_frames  # 活动图像列表
        self.uav_pose = uav_pose  # UAV位姿
        self.stitched_active_frames = stitched_active_frames  # 活动图像拼接结果
        self.all_frames=all_frames #所有帧对象

    # 计算活动图像的位姿
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
            if self.active_frames.frames[i].uv_pose is  None:
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



def load_images(image_dir, viewer, uav):
    img_count = 0
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".JPG"):
            img_count += 1
            # if img_count > 50:
            #     break
            path = os.path.join(image_dir, filename)
            uav.read_frame(path)
            viewer.add_image(path, uav.frames[-1].uv_pose * 9)
            viewer.set_camera_pos(uav.frames[-1].uv_pose * 9)


if __name__ == "__main__":
    image_dir = '/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/' 

    img_count = 0  
    uav=UAV_LOCATION()

    viewer = ImageViewer()
    
    load_thread = threading.Thread(target=load_images, args=(image_dir, viewer, uav))
    load_thread.start()

    while True:
        # 显示图像
        cv2.imshow('Image Viewer', viewer.view)
        # 退出条件
        if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
            break

    cv2.destroyAllWindows()
    # 等待加载线程结束
    load_thread.join()


