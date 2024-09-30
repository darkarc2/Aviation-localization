import cv2
import numpy as np
import os
import time
from Image_Viewer import ImageViewer
from Frame_class import Frame, Keypoint, ActiveImage
import threading
from modules.xfeat import XFeat
import tools


xfeat = XFeat()

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
        active_frames2=[frame for frame in self.all_frames if frame.is_active and frame.pyramid_images is not None]
        images = [frame.pyramid_images[0] for frame in active_frames2]
        keypoints_list = [frame.keypoints for frame in active_frames2]
    
        for i in range(1, len(images)):
            # 检查当前帧的uv_pose是否为None
            if active_frames2[i].uv_pose is  None:
                active_frames2[i].uv_pose= {'translation':  np.array([0, 0]),
                    'rotation': 0,
                    'Homography': None}
                # # # 获取当前图像和基准图像的特征点
                # keypoints1 = [kp for kp in keypoints_list[i-1] if kp.pyramid_layer == 2]
                # keypoints2 = [kp for kp in keypoints_list[i] if kp.pyramid_layer == 2]
                
                # # 使用BFMatcher进行特征点匹配
                # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # descriptors1 = np.array([kp.descriptor for kp in keypoints1])
                # descriptors2 = np.array([kp.descriptor for kp in keypoints2])
    
                # matches = bf.match(descriptors1, descriptors2)
                # matches = sorted(matches, key=lambda x: x.distance)
                
                # # 获取匹配点
                # points1 = [keypoints1[m.queryIdx].pt for m in matches]
                # points2 = [keypoints2[m.trainIdx].pt for m in matches]
                # points1 = np.array(points1)
                # points2 = np.array(points2)
                points1, points2 = xfeat.match_xfeat(images[i-1], images[i], top_k = 4000)
                # orb = cv2.ORB_create()
                # points1, descriptors1 = orb.detectAndCompute(images[i-1], None)
                # points2, descriptors2 = orb.detectAndCompute(images[i], None)
                # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # matches = matcher.match(descriptors1, descriptors2)
                # matches = sorted(matches, key=lambda x: x.distance)

                # points1 = np.array([points1[m.queryIdx].pt for m in matches[:50]])
                # points2 = np.array([points2[m.trainIdx].pt for m in matches[:50]])
                
                # 计算平移并进行图像拼接
                deta_pose = self.compute_translation_and_rotation(images[i-1], images[i], points1, points2)
                
                # 获取前一帧的旋转角度
                prev_rotation = active_frames2[i-1].uv_pose['rotation']*-1
                prev_rotation=np.deg2rad(prev_rotation )
                
                # 计算前一帧的旋转矩阵
                cos_theta = np.cos(prev_rotation)
                sin_theta = np.sin(prev_rotation)
                rotation_matrix = np.array([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])
                
                # 重新计算平移向量
                translation_global = rotation_matrix @ (deta_pose['translation']*-1)
                
                # 更新当前帧的平移和旋转
                active_frames2[i].uv_pose['translation'] = translation_global + active_frames2[i-1].uv_pose['translation']
                active_frames2[i].uv_pose['rotation'] = deta_pose['rotation']*-1  + active_frames2[i-1].uv_pose['rotation']
                print(f"帧{active_frames2[i].object_id}的uv_pose为{active_frames2[i].uv_pose}")
    def extract_pose_from_homography(self,H):
        # Camera intrinsic matrix (assuming fx = fy = 1, cx = cy = 0 for simplicity)
        # K = np.eye(3)
        #  # 分解单应矩阵 H
        a=3976/35.9
        b=2652/24
        f=35

        K = np.array([
            [f*a, 0, 3976//2],
            [0, f*b, 2652//2],
            [0, 0, 1]
        ])

        # Decompose the homography matrix
        _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

        # Select the first solution (you may need to validate which solution is correct)
        R = Rs[0]
        T = Ts[0]
        T*=412.24953990779335
        # T*=3742.9108374078

        # Calculate rotation angle (in degrees)
        theta = np.arctan2(R[1, 0], R[0, 0]) * (180.0 / np.pi)

        # Calculate translation (in pixels)
        tx, ty = T[0], T[1]

        return theta, tx, ty
    # 计算特征点匹配的平移向量和旋转矩阵
    def compute_translation_and_rotation(self, img_1, img_2, points1, points2):
        # 将匹配点转换为浮点数并重塑为二维数组
        # # Calculate the Homography matrix，计算单应性矩阵
        H, mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        H_inv, mask2 = cv2.findHomography(points2,points1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        theta, tx, ty=self.extract_pose_from_homography(H_inv)
        distance =(tx**2+ty**2)**0.5
        # print("相对位移：",distance)
        # print(tx,ty)
        # self.all_frames[-1].pose=np.array([tx,ty])+self.all_frames[-2].pose

        h, w = img_1.shape[:2]
        corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

        # 使用 mask 筛选出匹配的特征点
        matched_keypoints1 = [points1[i] for i in range(len(mask)) if mask[i]]
        matched_keypoints2 = [points2[i] for i in range(len(mask)) if mask[i]]
        matched_keypoints1 = np.array(matched_keypoints1)
        matched_keypoints2 = np.array(matched_keypoints2)


        # opencv显示匹配结果
        # matches = [cv2.DMatch(i, i, 0) for i in range(len(matched_keypoints1))]
        # keypoints1 = [cv2.KeyPoint(x=point[0], y=point[1], size=1) for point in matched_keypoints1]
        # keypoints2 = [cv2.KeyPoint(x=point[0], y=point[1], size=1) for point in matched_keypoints2]
        # img_matches = cv2.drawMatches(img_1, keypoints1, img_2, keypoints2, matches, None)
        # cv2.imshow('Matches', img_matches)
        # cv2.waitKey(0)
        # matched_keypoints1 = points1
        # matched_keypoints2 = points2
        # print("相对位移4：",np.mean(matched_keypoints1-matched_keypoints2,axis=0))
        center_p=np.mean(matched_keypoints2-matched_keypoints1,axis=0)


        # 计算质心
        centroid1 = np.mean(matched_keypoints1, axis=0)
        centroid2 = np.mean(matched_keypoints2, axis=0)

        # 去中心化
        centered_keypoints1 = matched_keypoints1 - centroid1
        centered_keypoints2 = matched_keypoints2 - centroid2

        # 计算协方差矩阵
        H = np.dot(centered_keypoints1.T, centered_keypoints2)

        # 进行 SVD 分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = np.dot(Vt.T, U.T)

        # 确保旋转矩阵是一个正交矩阵
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # # 计算位移向量
        # center_p = centroid2 - np.dot(R, centroid1)

        # print("旋转矩阵 R:")
        # print(R)
        #角度
        theta = np.arctan2(R[1, 0], R[0, 0])
        theta = np.rad2deg(theta)
        # theta=0.3

        
        # 保存平移向量和旋转角度到pose变量
        deta_pose = {
            'translation': center_p,
            'rotation': theta,
            'Homography': H
        }
        
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
    jump=0
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".JPG"):
            # jump+=1
            # if jump<=0:
            #     continue
            img_count += 1
            if img_count > 93:
                break
            path = os.path.join(image_dir, filename)
            uav.read_frame(path)
            # viewer.add_image(path, uav.frames[-1].uv_pose * 9)
            for i in range(len(uav.frames)):
                if uav.frames[len(uav.frames)-i-1].uv_pose is not None:
                    viewer.set_camera_pos(uav.frames[len(uav.frames)-i-1].uv_pose['translation'].copy())
                    break
            # viewer.set_camera_pos(uav.frames[-1].uv_pose.copy())



def print_data(uav):
    csv_file_path = '/home/arc/works/review_prj/UAV_slam/src/03.csv'
    frames = tools.read_csv(csv_file_path)
    
    # 假设uav对象已经定义并包含frames
    errors = tools.calculate_errors(frames, uav)
    
    for error in errors:
        print(f"帧{error['frame_num']} - 纬度误差: {error['lat_error']}, 经度误差: {error['lon_error']}, 距离误差: {error['distance_error']}米")


if __name__ == "__main__":
    image_dir = 'D:/Documents/CodeProjects/Aviation-localization/src/' 
    image_dir = 'D:/BaiduNetdiskDownload/UAV_data' 
    image_dir = "/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone"
    # image_dir = "/home/arc/works/review_prj/UAV_slam/bad_imgs"


    img_count = 0  
    uav=UAV_LOCATION()

    viewer = ImageViewer(uav)
    
    load_thread = threading.Thread(target=load_images, args=(image_dir, viewer, uav))
    load_thread.start()

    while True:
        # 显示图像
        viewer.run()
        cv2.imshow('Image Viewer', viewer.view)
        # 退出条件
        if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
            break

    cv2.destroyAllWindows()
    # 等待加载线程结束
    load_thread.join()

    t_pose=uav.frames[-1].get_pose()
    print("utm_POS",t_pose)

    # for i in range(len(uav.frames)):
    #     print(f"帧{i}pose_in_camera:{uav.frames[i].uv_pose_to_pose_in_camera()}")
    # for i in range(len(uav.frames)):
    #     print(f"帧{i}pose_in_camera:{uav.frames[i].get_pose()}")
    for i in range(len(uav.frames)):
        print(f"帧{i}pose_in_latlon:{tools.utm_to_latlon(uav.frames[i].get_pose()[0],uav.frames[i].get_pose()[1])}")

    
    print(tools.utm_to_latlon(t_pose[0],t_pose[1]))
    # for i in range(len(uav.frames)):
    #     print(f"帧{i}经纬度:{tools.utm_to_latlon(uav.frames[i].get_pose()[0],uav.frames[i].get_pose()[1])}")
    print_data(uav)



