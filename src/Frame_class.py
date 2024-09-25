import cv2
import numpy as np

class Frame:
    _id_counter = 0  # 类变量，用于计数
    def __init__(self, image_path):
        self.image_path = image_path  # 对应图像文件路径

        self.image = cv2.imread(image_path)  # 读取图像

        self.pyramid_images = self.generate_pyramid(self.image)  # 生成金字塔图像
        self.keypoints = []  # 特征点列表
        self.pose = None  # 位姿
        self.pose_in_camera = None  # 位姿
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
    def uv_pose_to_pose_in_camera(self,uv_pose):
        a=7952/35.9
        b=5304/24
        f=35

        K = np.array([
            [f*a, 0, 3976],
            [0, f*b, 2652],
            [0, 0, 1]
        ])

        # 相机坐标系下各图片中心点
        pose_in_camera=uv_pose+np.array([3976,2652]) #把像素坐标转到图片中心下的坐标
        pose_in_camera=np.array([pose_in_camera[0],pose_in_camera[1],1]) #转为齐次坐标

        k_inv=np.linalg.inv(K)
        pose_in_camera=466*k_inv@(pose_in_camera)

        return pose_in_camera

    def get_pose(self):
        # 把第一张图片的位置作为原点，可以得到t
        # 结合第二张图片的位置，可以得到R
        R=np.array([[-0.48546517,-0.83104347,-0.27145961],
            [ 0.85030412,-0.52101022, 0.07437244],
            [-0.20323996,-0.19471799, 0.959572  ]])
        t=np.array([ 772767.90754028,3577889.15665317, 0])

        pose_in_camera=self.uv_pose_to_pose_in_camera(self.uv_pose)
        self.pose=R@(pose_in_camera)+t #计算UTM坐标下的飞机位姿
        return self.pose

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
        self.activate_len=15

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
        # self.frames.remove(frame) #活动帧中移除
    def reactivate_frame(self, frame):
        frame.is_active=True
        frame.image = cv2.imread(frame.image_path)  # 读取图像
        frame.pyramid_images = frame.generate_pyramid(frame.image)  # 生成金字塔图像


    def get_active_frame_ids(self):
        return [frame.object_id for frame in self.frames]