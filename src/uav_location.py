import cv2
import numpy as np
import os
import time
from Image_Viewer import ImageViewer
from Frame_class import Frame, ActiveImage
import threading
from accelerated_features.modules.xfeat import XFeat
import tools    
from wildnav.src.Global_position import GeoLocator

xfeat = XFeat()

class GlobalMap:
    def __init__(self, active_frames,all_frames,ref_map_path, ref_map_csv,observation_frame_id=None,  uav_pose=None):
        # self.observation_frame_id = observation_frame_id  # 观察框所在图像ID
        # self.active_frames = active_frames  # 活动图像列表
        # self.uav_pose = uav_pose  # UAV位姿
        self.all_frames=all_frames #所有帧对象
        # 初始化 GeoLocator 使用新的子地图
        self.geo_locator = GeoLocator(ref_map_path, ref_map_csv)
    

    # 计算特征点匹配的平移向量和旋转矩阵
    def compute_translation_and_rotation(self, img_1, img_2, points1, points2):
        # 将匹配点转换为浮点数并重塑为二维数组
        # # Calculate the Homography matrix，计算单应性矩阵
        H, mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)



        h, w = img_1.shape[:2]
        corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

        # 使用 mask 筛选出匹配的特征点
        matched_keypoints1 = [points1[i] for i in range(len(mask)) if mask[i]]
        matched_keypoints2 = [points2[i] for i in range(len(mask)) if mask[i]]
        matched_keypoints1 = np.array(matched_keypoints1)
        matched_keypoints2 = np.array(matched_keypoints2)


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
    # 计算最新帧的图像的位姿
    def compute_translation(self):


        new_id=len(self.all_frames)-1
        now_frame=self.all_frames[new_id]
        last_frame=self.all_frames[new_id-1]
        now_frame.uv_pose= {'translation':  np.array([0, 0]),
            'rotation': 0,
            'Homography': None}
        

        # 法1===============
        # points1, points2 = xfeat.match_xfeat(last_frame.image, now_frame.image, top_k = 4000)


        # 法2===============
        # orb = cv2.ORB_create()
        # points1, descriptors1 = orb.detectAndCompute(last_frame.image, None)
        # points2, descriptors2 = orb.detectAndCompute(now_frame.image, None)
        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = matcher.match(descriptors1, descriptors2)
        # matches = sorted(matches, key=lambda x: x.distance)

        # points1 = np.array([points1[m.queryIdx].pt for m in matches[:50]])
        # points2 = np.array([points2[m.trainIdx].pt for m in matches[:50]])
                
        # 法3===============
        output0 = xfeat.detectAndCompute(last_frame.image, top_k = 8000)[0]
        output1 = xfeat.detectAndCompute(now_frame.image, top_k = 8000)[0]

        #Update with image resolution (required)
        output0.update({'image_size': (last_frame.image.shape[1], last_frame.image.shape[0])})
        output1.update({'image_size': (now_frame.image.shape[1], now_frame.image.shape[0])})
        points1, points2 = xfeat.match_lighterglue(output0, output1)
        # 计算平移并进行图像拼接
        deta_pose = self.compute_translation_and_rotation(last_frame.image, now_frame.image, points1, points2)
        
        # 获取前一帧的旋转角度
        prev_rotation = last_frame.uv_pose['rotation']*-1
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
        now_frame.uv_pose['translation'] = translation_global + last_frame.uv_pose['translation']
        now_frame.uv_pose['rotation'] = deta_pose['rotation']*-1  + last_frame.uv_pose['rotation']
        print(f"帧{now_frame.object_id}的uv_pose为{now_frame.uv_pose}")

class UAV_LOCATION:

    def __init__(self, ref_map_path, ref_map_csv):
        self.frames = []
        self.active_frames = ActiveImage()  
        self.map=GlobalMap(self.active_frames,self.frames,ref_map_path, ref_map_csv)
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

import queue
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)
# 创建一个队列来存储需要更新的帧信息
update_queue = multiprocessing.Queue()
update_lock = multiprocessing.Lock()
process_lock = multiprocessing.Lock()
flag_is_finish = True

def locate_image_in_background(uav, suspected_pose, delta_pose, start_pose_utm, frame_index,queue):
    flag_is_finish = False
    location = uav.map.geo_locator.locate_image(uav.frames[frame_index].image, suspected_pose, delta_pose)
    if location is not None:
        loca_utm = np.array(tools.latlon_to_utm(location[0], location[1])) - start_pose_utm
        queue.put((frame_index, loca_utm))
        print(f"帧{frame_index}定位完成, add{(frame_index, loca_utm)}")
        print("===========后台修正完成=====================") 
    else:
        print(f"帧{frame_index}绝对定位失败!!!!!!!!!!error!!!!!!!!!")
    flag_is_finish = True


def update_frames(uav):
    while not update_queue.empty():
        frame_index, loca_utm = update_queue.get()
        # 获取修正帧的原始位置
        corrected_frame_pose = uav.frames[frame_index].get_pose()
        # 计算偏移量
        offset = loca_utm - corrected_frame_pose

        # 更新从修正帧到最新帧的所有帧的位置信息
        for i in range(frame_index, len(uav.frames)):
            current_pose = uav.frames[i].get_pose()
            new_pose = current_pose + offset
            uav.frames[i].set_pose(new_pose)
            print(f'帧{i}更新修正ing....after:{uav.frames[i].uv_pose}')

csv_file_path = '/home/arc/works/review_prj/UAV_slam/src/03.csv'
true_pose = tools.get_true_pose(csv_file_path)
def load_images(image_dir, viewer, uav):
    img_count = 0
    jump=0
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".JPG"):
            # jump+=1
            # if jump<=300:
            #     continue
            img_count += 1
            if img_count > 90:
                break
            path = os.path.join(image_dir, filename)
            uav.read_frame(path)
            update_frames(uav)


            if img_count % 15==0:
                start_pose_utm = [true_pose[jump][2],true_pose[jump][3]]  # 第一帧的utm坐标真值，后续都要基于这个初始真值计算
                new_pose=uav.frames[-1].get_pose()+start_pose_utm   # 当前帧的utm坐标
                suspected_pose=list(tools.utm_to_latlon(new_pose[0],new_pose[1]))  # 当前帧的经纬度
                delta_pose = 0.005

                frame_index = len(uav.frames) - 1
                # if thread_lock.acquire(blocking=False):
                # threading.Thread(target=locate_image_in_background, args=(uav, suspected_pose, delta_pose, start_pose_utm, frame_index)).start()
                # try:
                #     if process_lock.acquire(False):  # 非阻塞锁定
                if flag_is_finish:
                        multiprocessing.Process(target=locate_image_in_background, args=(uav, suspected_pose, delta_pose, start_pose_utm, frame_index,update_queue)).start()

                # location=uav.map.geo_locator.locate_image(uav.frames[-1].image,suspected_pose, delta_pose)
                # if location is not None:
                    
                #     loca_utm=np.array(tools.latlon_to_utm(location[0],location[1]))-start_pose_utm

                #     uav.frames[-1].set_pose(loca_utm)
                #     print(f'after:{uav.frames[-1].uv_pose}')


                #     start_pose_utm = [true_pose[jump][2],true_pose[jump][3]]  # 第一帧的utm坐标真值，后续都要基于这个初始真值计算
                #     new_pose=uav.frames[-1].get_pose()+start_pose_utm   # 当前帧的utm坐标
                #     now_frame=int(filename[-8:-4])                      # 当前帧的文件名
                #     now_true_pose = [true_pose[now_frame-1][2],true_pose[now_frame-1][3]]   #根据当前文件名获取真值
                #     distance_error = ((now_true_pose[0] - new_pose[0]) ** 2 + (now_true_pose[1] - new_pose[1]) ** 2) ** 0.5  # 计算距离误差
                #     print(f"帧{filename} -, 距离误差: {distance_error}米")

            for i in range(len(uav.frames)):  # 从最新帧开始显示
                if uav.frames[len(uav.frames)-i-1].uv_pose is not None:
                    viewer.set_camera_pos(uav.frames[len(uav.frames)-i-1].uv_pose['translation'].copy())
                    break




def print_data(uav):
    
    frames = tools.read_csv(csv_file_path)
    true_pose = tools.get_true_pose(csv_file_path)
    
    # 假设uav对象已经定义并包含frames
    errors = tools.calculate_errors(frames, uav)
    
    for error in errors:
        print(f"帧{error['frame_num']} - 纬度误差: {error['lat_error']}, 经度误差: {error['lon_error']}, 距离误差: {error['distance_error']}米")


if __name__ == "__main__":
    image_dir = 'D:/Documents/CodeProjects/Aviation-localization/src/' 
    image_dir = 'D:/BaiduNetdiskDownload/UAV_data' 
    image_dir = "/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone"
    # image_dir = "/home/arc/works/review_prj/UAV_slam/bad_imgs"
    ref_img_path = "/home/arc/works/review_prj/UAV_slam/src/wildnav/assets/map/"
    ref_map_csv = "/home/arc/works/review_prj/UAV_slam/src/wildnav/assets/map/map.csv"


    img_count = 0  
    uav=UAV_LOCATION(ref_img_path,ref_map_csv)

    viewer = ImageViewer(uav)
    
    # load_thread = threading.Thread(target=load_images, args=(image_dir, viewer, uav))
    # load_thread.start()


    # 等待加载线程结束
    # load_thread.join()
    load_images(image_dir, viewer, uav)

    t_pose=uav.frames[-1].get_pose()
    print("utm_POS",t_pose)

    while True:
        # 显示图像
        viewer.run()
        cv2.imshow('Image Viewer', viewer.view)
        # 退出条件
        if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
            break
    cv2.destroyAllWindows()
    # for i in range(len(uav.frames)):
    #     print(f"帧{i}pose_in_camera:{uav.frames[i].uv_pose_to_pose_in_camera()}")
    # for i in range(len(uav.frames)):
    #     print(f"帧{i}pose_in_camera:{uav.frames[i].get_pose()}")
    # for i in range(len(uav.frames)):
    #     print(f"帧{i}pose_in_latlon:{tools.utm_to_latlon(uav.frames[i].get_pose()[0],uav.frames[i].get_pose()[1])}")

    
    print(tools.utm_to_latlon(t_pose[0],t_pose[1]))
    # for i in range(len(uav.frames)):
    #     print(f"帧{i}经纬度:{tools.utm_to_latlon(uav.frames[i].get_pose()[0],uav.frames[i].get_pose()[1])}")
    print_data(uav)



