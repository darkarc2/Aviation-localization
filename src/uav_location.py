import cv2
import numpy as np
import os
import time

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
        # if self.object_id == 1:
        #     self.uv_pose = np.array([-161.0696106,-865.79919434])

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


    def draw_lines_between_centers(self, image):
        # 获取每个帧的中心点
        centers = [frame.uv_pose for frame in self.active_frames.frames if frame.uv_pose is not None]
        # 将中心点转换为整数坐标
        centers = [(int(center[0]), int(center[1])) for center in centers]
        # 画连线
        for i in range(1, len(centers)):
            cv2.line(image, centers[i-1], centers[i], (0, 0, 255), 20)
        return image

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
        
        # 进行图像拼接
        result=self.translation_and_warp(img_1, img_2, translation_vector)

        # 保存平移向量到pose变量
        deta_pose = translation_vector
        
        return result, deta_pose
    
    # 基于detapose图像平移和拼接
    def translation_and_warp(self, img_1, img_2,deta_pose):
        # 计算平移矩阵
        M = np.array([[1, 0, deta_pose[0]], [0, 1, deta_pose[1]]], dtype=np.float32)
        
        # 获取图像尺寸
        h1, w1 = img_1.shape[:2]
        h2, w2 = img_2.shape[:2]
        
        # 计算拼接后的图像尺寸
        x_min = min(0, deta_pose[0])
        y_min = min(0, deta_pose[1])
        x_max = max(w1, w2 + deta_pose[0])
        y_max = max(h1, h2 + deta_pose[1])
        
        # 创建一个足够大的画布
        canvas_width = int(x_max - x_min)
        canvas_height = int(y_max - y_min)
        result = np.zeros((canvas_height, canvas_width, 3), dtype=img_1.dtype)
        
        # 将第一张图片放到画布上
        result[-int(y_min):h1-int(y_min), -int(x_min):w1-int(x_min)] = img_1
        
        # 将第二张图片平移后放到画布上
        result[int(deta_pose[1])-int(y_min):int(deta_pose[1])-int(y_min)+h2, int(deta_pose[0])-int(x_min):int(deta_pose[0])-int(x_min)+w2] = img_2
        
        return result
    
    def stitch_active_frames(self):
        # 拼接活动图像的逻辑
        if not self.active_frames or len(self.active_frames.frames) == 0:
            raise ValueError("No active images to stitch.")
        if len(self.active_frames.frames)==1:
            return self.active_frames.frames[0].pyramid_images[2]
        # 获取所有活动帧的图像
        images = [frame.pyramid_images[2] for frame in self.active_frames.frames]
        keypoints_list = [frame.keypoints for frame in self.active_frames.frames]
    
        # print(f"正在拼接{len(images)}张图像:开始帧{self.active_frames.frames[0].object_id}，结束帧{self.active_frames.frames[-1].object_id}")
        poses = []
        global_map=None
        for i in range(1, len(images)):
            # 检查当前帧的uv_pose是否为None
            if self.active_frames.frames[i].uv_pose is not None:
                deta_pose = self.active_frames.frames[i].uv_pose-self.active_frames.frames[i-1].uv_pose
                new_image = self.translation_and_warp(images[i-1], images[i], deta_pose)
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
                new_image, deta_pose = self.compute_translation_and_warp(images[i-1], images[i], points1, points2)
                self.active_frames.frames[i].uv_pose = deta_pose + self.active_frames.frames[i-1].uv_pose
            if global_map is None:
                global_map = new_image
            else:
                # deta_pose = self.active_frames.frames[i].uv_pose-self.active_frames.frames[i-1].uv_pose
                global_map = self.translation_and_warp(global_map, new_image, deta_pose)
        
        # 更新reduced_image属性
        self.reduced_image = global_map
        self.poses = poses
         # 在拼接后的图像上画连线
        self.reduced_image = self.draw_lines_between_centers(self.reduced_image)

        return self.reduced_image
    

    def dynamic_stitch_based_on_observation(self):

         # 开始计时
        start_time = time.time()

        if self.observation_frame_id is None:
            raise ValueError("Observation frame ID is not set.")
        

        before_observation_flag=True
        activate_count=0
        if len(self.all_frames)==0: #如果没有帧，直接返回
            return self.stitch_active_frames()
        for frame in self.all_frames:
            if frame.object_id == self.observation_frame_id:
                before_observation_flag=False # 找到观察框所在帧，后续帧需要激活
            if before_observation_flag :
                if frame.is_active==True:
                    self.active_frames.remove_frame(frame)  # 移除观察框之前的活动帧
            elif activate_count<=self.active_frames.activate_len: #控制激活的帧数
                if frame.is_active==False:
                    self.active_frames.reactivate_frame(frame) # 重新激活观察框之后的帧
                    self.active_frames.add_frame(frame) # 添加到活动帧列表
                activate_count+=1 # 计数已激活的帧数
            else:
                break

        stitched_image = self.stitch_active_frames()  # 拼接活动帧

        # 结束计时
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"dynamic_stitch_based_on_observation 耗时: {elapsed_time:.4f} 秒")

        
        return stitched_image



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
        # self.map_img=self.map.stitch_active_frames()

        # 时刻取最新的地图，观察窗应该从最新帧id-活动地图帧数量开始
        self.map.observation_frame_id = max(0,self.frames[-1].object_id-self.active_frames.activate_len+1)
        self.map_img=self.map.dynamic_stitch_based_on_observation()

    
    def get_global_map(self):
        return self.map_img
    
    def get_map_observaion_id(self):
        return self.map.observation_frame_id
    
    def set_map_observaion_id(self,observation_frame_id):
        self.map.observation_frame_id=observation_frame_id


def show_frames(frames):
    for frame in frames:
        cv2.imshow("Frame", frame.pyramid_images[-1])
        cv2.waitKey(0)
    return



# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['dragging'] = True
        param['start_x'] = x
        param['start_y'] = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['dragging']:
            pass

    elif event == cv2.EVENT_LBUTTONUP:
        param['dragging'] = False
        dx = x - param['start_x']
        dy = y - param['start_y']
        param['start_x'] = x
        param['start_y'] = y
        if abs(dy) < 50:
            dy = 0
        
        # 更新观察框的位置
        change_flag=True
        if dy > 0:
            param['uav'].map.observation_frame_id += 1
            if param['uav'].map.observation_frame_id >= len(param['uav'].frames)-param['uav'].active_frames.activate_len:
                param['uav'].map.observation_frame_id = param['uav'].map.observation_frame_id - 1
                change_flag=False
        elif dy < 0:
            param['uav'].map.observation_frame_id -= 1
            if param['uav'].map.observation_frame_id < 0:
                param['uav'].map.observation_frame_id = 0
                change_flag=False
        else:
            change_flag=False

        if change_flag:
            # 动态加载附近的活动帧
            stitched_image = param['uav'].map.dynamic_stitch_based_on_observation()
            # 显示更新后的地图
            cv2.imshow("g_map", stitched_image)


if __name__ == "__main__":
    image_dir = '/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/' 

    img_count = 0  
    uav=UAV_LOCATION()

    # 设置初始的观察框位置
    observation_frame_id = 0
    uav.set_map_observaion_id(observation_frame_id)

     # 初始化鼠标事件参数
    mouse_params = {'dragging': False, 'start_x': 0, 'start_y': 0,'uav':uav}

    # 创建窗口并设置鼠标事件回调函数
    cv2.namedWindow("g_map")
    cv2.setMouseCallback("g_map", mouse_callback, mouse_params)
    
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".JPG"):
            img_count+=1
            if img_count>20:
                break


            path=os.path.join(image_dir, filename)
            uav.read_frame(path)
            global_map=uav.get_global_map()
            # cv2.imshow("g_map",global_map)
            # cv2.waitKey(0)


            # 显示初始的全局地图
            # stitched_image = uav.map.dynamic_stitch_based_on_observation()
            cv2.imshow("g_map", global_map)
            cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


