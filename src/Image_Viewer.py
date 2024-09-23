import cv2
import numpy as np
import os
import random
import threading


class ImageViewer:
    def __init__(self,uav, window_size=(800, 600), zoom=1.0):

        self.window_size = window_size
        self.zoom = zoom
        self.camera_pos = np.array([0, 0], dtype=np.float32)
        self.dragging = False
        self.last_mouse_pos = np.array([0, 0])
        self.frames = uav.frames
        self.uav=uav
        self.view = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.mouse_callback)

    def set_camera_pos(self,pos):
        self.camera_pos = pos
        # self.run()

    # 鼠标回调函数，用于处理鼠标事件
    def mouse_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下事件
            self.dragging = True
            self.last_mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:  # 鼠标移动事件
            delta = (np.array([x, y]) - self.last_mouse_pos) / self.zoom
            self.camera_pos -= delta
            self.last_mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_LBUTTONUP:  # 左键抬起事件
            self.dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:  # 鼠标滚轮事件
            center_before_zoom = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / (2 * self.zoom)
            if flags > 0:
                self.zoom *= 2
                if self.zoom > 1:
                    self.zoom = 1
            else:
                self.zoom /= 1.1
            center_after_zoom = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / (2 * self.zoom)
            self.camera_pos += center_before_zoom - center_after_zoom

    # def add_image(self, path, pos):
    #     self.frames.append({'path': path, 'pos': pos, 'is_active': False, 'img': None})

    def run(self):
        # while True:
        
            # 计算相机视图范围
            top_left = self.camera_pos
            bottom_right = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_top_left = top_left - np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_bottom_right = bottom_right + np.array([self.window_size[0], self.window_size[1]]) / self.zoom
    
            # 动态加载和卸载图像
            for frame in self.frames:
                if frame.uv_pose is None:
                    continue
                pos = frame.uv_pose.copy()
                img_top_left = pos
                img_bottom_right = pos + np.array([3000, 3000])  # 假设图像大小为1000x1000
    
                # 判断图像是否在视图范围内
                if (img_bottom_right[0] > extended_top_left[0] and img_top_left[0] < extended_bottom_right[0] and
                    img_bottom_right[1] > extended_top_left[1] and img_top_left[1] < extended_bottom_right[1]):
                    if not frame.is_active:
                        # frame.image = cv2.imread(frame.image_path)  # 加载图像
                        # frame.is_active = True
                        print('Load image:', frame.image_path)
                        self.uav.active_frames.reactivate_frame(frame)
                else:
                    if frame.is_active:
                        # frame.image = None  # 卸载图像
                        # frame.is_active = False
                        self.uav.active_frames.remove_frame(frame)
    
            # 创建黑色背景
            self.view.fill(0)
            # 渲染活动图像
            for frame in self.frames:
                if frame.is_active and frame.uv_pose is not None:
                    img = frame.image
                    pos = frame.uv_pose.copy()
                    img_top_left = pos
                    img_bottom_right = pos + np.array([img.shape[1], img.shape[0]])
    
                    # 判断图像是否在视图范围内
                    if (img_bottom_right[0] > top_left[0] and img_top_left[0] < bottom_right[0] and
                        img_bottom_right[1] > top_left[1] and img_top_left[1] < bottom_right[1]):
                        view_pos = (pos - top_left) * self.zoom
                        view_pos = view_pos.astype(int)
    
                        x1 = max(0, view_pos[0])
                        y1 = max(0, view_pos[1])
                        x2 = min(self.window_size[0], view_pos[0] + int(img.shape[1] * self.zoom))
                        y2 = min(self.window_size[1], view_pos[1] + int(img.shape[0] * self.zoom))
    
                        img_x1 = max(0, -view_pos[0])
                        img_y1 = max(0, -view_pos[1])
                        img_x2 = img_x1 + (x2 - x1)
                        img_y2 = img_y1 + (y2 - y1)
    
                        img_resized = cv2.resize(img, (int(img.shape[1] * self.zoom), int(img.shape[0] * self.zoom)))
    
                        self.view[y1:y2, x1:x2] = img_resized[img_y1:img_y2, img_x1:img_x2]
    
            # 在图片之间绘制中线点连线
            for i in range(1, len(self.frames)):
                if self.frames[i-1].is_active and self.frames[i].is_active and self.frames[i].uv_pose is not None :
                    pos1 = self.frames[i-1].uv_pose.copy()
                    pos2 = self.frames[i].uv_pose.copy()
    
                    # 计算中线点
                    mid_point1 = pos1 + np.array([self.frames[i-1].image.shape[1] // 2, self.frames[i-1].image.shape[0] // 2])
                    mid_point2 = pos2 + np.array([self.frames[i].image.shape[1] // 2, self.frames[i].image.shape[0] // 2])
    
                    # 转换到视图坐标
                    view_mid_point1 = (mid_point1 - top_left) * self.zoom
                    view_mid_point2 = (mid_point2 - top_left) * self.zoom
    
                    view_mid_point1 = view_mid_point1.astype(int)
                    view_mid_point2 = view_mid_point2.astype(int)
    
                    # 绘制连线
                    cv2.line(self.view, tuple(view_mid_point1), tuple(view_mid_point2), (0, 255, 0), 2)
    
            # 添加相机位置文本
            cv2.putText(self.view, f'Camera Pos: {self.camera_pos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
            # 添加相机位置文本
            cv2.putText(self.view, "press esc to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # def show_trajectory(self):
    #     # 绘制轨迹
    #     for i in range(1, len(self.frames)):
    #         cv2.line(self.view, self.frames[i-1]["pos"], self.frames[i]["pos"], (0, 255, 0), 2)

    # def show(self):

    #     while True:
    #         # 显示图像
    #         cv2.imshow('Image Viewer', self.view)
    #         # 退出条件
    #         if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
    #             break
    #     cv2.destroyAllWindows()
    # def start(self):
    #     t=threading.Thread(target=self.show)
    #     t.start()



if __name__ == "__main__":

    viewer = ImageViewer()

    viewer.add_image("/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG", np.array([0, 0]) )
    viewer.add_image("/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG", np.array([-19.02805901*9,-98.53948212*9]) )
    
    viewer.start()
