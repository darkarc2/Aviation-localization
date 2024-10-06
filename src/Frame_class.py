import cv2
import numpy as np

class Frame:
    _id_counter = 0  # 类变量，用于计数
    def __init__(self, image_path):
        self.image_path = image_path  # 对应图像文件路径

        self.image = cv2.imread(image_path)  # 读取图像

        self.pyramid_images = self.generate_pyramid(self.image)  # 生成金字塔图像
        self.keypoints = []  # 特征点列表
        self.pose = None  # utm下的位姿
        self.pose_in_camera = None  # 相机坐标系下位姿
        self.uv_pose = None  # 位姿在图像上的像素坐标，这里是在最高分辨率的图像上的像素坐标
        self.is_active = True  # 是否是活帧
        self.object_id = Frame._id_counter# 对象ID
        Frame._id_counter += 1

        # # 检测每层金字塔下的特征点
        # for layer in range(len(self.pyramid_images)):
        #     self.detect_keypoints_at_layer(layer)

        # 第一张图片的位姿初始化
        if self.object_id == 0:
            self.uv_pose = {'translation':  np.array([0, 0]),
                    'rotation': 0}


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

    # # 在指定层检测特征点
    # def detect_keypoints_at_layer(self, layer):
    #     if self.pyramid_images is None or layer >= len(self.pyramid_images):
    #         raise ValueError("Invalid pyramid layer.")
        
    #     orb = cv2.ORB_create()
    #     keypoints, descriptors = orb.detectAndCompute(self.pyramid_images[layer], None)
        
    #     for kp, desc in zip(keypoints, descriptors):
    #         self.keypoints.append(Keypoint(kp.pt, layer, self, desc))
    #     return self.keypoints


    def update_uv_pose(self, uav_pose):
        # 更新位姿在图像上的像素坐标的逻辑
        self.uv_pose = uav_pose
    def get_pose_in_camera(self):
        a=3976/35.9
        b=2652/24
        f=35

        K = np.array([
            [f*a, 0, 3976//2],
            [0, f*b, 2652//2],
            [0, 0, 1]
        ])

        # 相机坐标系下各图片中心点
        pose_in_camera=self.uv_pose['translation'] #把像素坐标转到图片中心下的坐标
        pose_in_camera=np.array([pose_in_camera[0],pose_in_camera[1],1]) #转为齐次坐标

        k_inv=np.linalg.inv(K)
        pose_in_camera=466*k_inv@(pose_in_camera)

        return pose_in_camera

    def get_pose(self):
        #像素坐标转为utm坐标公式为P'=(RP+t)k,k为尺度变化因子,P'=kRP+kt

        #这里我直接通过拟合得到的参数进行转换，原理是一样的
        u=self.uv_pose['translation'][0]
        v=self.uv_pose['translation'][1]
        # a,b,c,d=-0.024339612852506524, 0.09639987552971423,0.02283516200259328, -0.06977090383329672
        # 2线路
        a,b,c,d=0.01617518940474838, 0.0924426899291575,-0.01170275523607696, -0.0668823186811507
        # 3线路
        # a,b,c,d=-0.16923672929767528, 0.12093727164720008,0.1324161747470498, -0.09062548226614338
        e,f=768562.1758137619, 3580349.1592050353   #ef其实就是位移矩阵，这里我先不添加了，后面要根据初始帧的utm坐标为基准来添加
        x = a * u + b * v  #+e
        y = c * u + d * v  #+f

        self.pose=np.array([x,y])
        return self.pose
    def set_pose(self,pose):
        # UTM坐标转为像素坐标公式为P=(kR)^(-1)(P'-kt),k为尺度变化因子
        # 这里我们直接使用拟合得到的参数进行转换，原理是一样的
        x = pose[0]
        y = pose[1]
        # 1线路
        # a, b, c, d = -0.024339612852506524, 0.09639987552971423, 0.02283516200259328, -0.06977090383329672
        # 2线路
        a,b,c,d=0.01617518940474838, 0.0924426899291575,-0.01170275523607696, -0.0668823186811507
        # 3线路
        # a,b,c,d=-0.16923672929767528, 0.12093727164720008,0.1324161747470498, -0.09062548226614338
        e, f = 768562.1758137619, 3580349.1592050353  # ef其实就是位移矩阵，这里我先不添加了，后面要根据初始帧的utm坐标为基准来添加

        # 计算逆矩阵
        det = a * d - b * c
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det

        # 计算像素坐标
        u = inv_a * x + inv_b * y  # - e
        v = inv_c * x + inv_d * y  # - f

        self.uv_pose['translation'][0] = u
        self.uv_pose['translation'][1] = v
        return self.uv_pose['translation']


        # 移除帧
    def remove_frame(self, frame):
        frame.is_active=False
        frame.image = None  # 删除图像
        frame.pyramid_images = None  # 删除图像
        # self.frames.remove(frame) #活动帧中移除
    # 重新激活帧
    def reactivate_frame(self, frame):
        frame.is_active=True
        frame.image = cv2.imread(frame.image_path)  # 读取图像
        frame.pyramid_images = frame.generate_pyramid(frame.image)  # 生成金字塔图像


# class Keypoint:
#     def __init__(self, pt, pyramid_layer, associated_image, descriptor, is_anomalous=False):
#         self.pt = pt  # 特征点像素位置
#         self.descriptor = descriptor  # 描述符
#         self.pyramid_layer = pyramid_layer  # 特征点所在图像金字塔层
#         self.associated_image = associated_image  # 关联图像
#         self.is_anomalous = is_anomalous  # 是否为异常点

#     def is_within_bounds(self, image_shape):
#         x, y = self.pt
#         return 0 <= x < image_shape[1] and 0 <= y < image_shape[0]

#     def mark_anomalous(self):
#         self.is_anomalous = True

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



    # 获取活动帧的 object_id 列表
    def get_active_frame_ids(self):
        return [frame.object_id for frame in self.frames]