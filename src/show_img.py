import cv2
import numpy as np
import os
import random



class ImageViewer:
    def __init__(self, window_size=(800, 600), zoom=1.0):

        self.window_size = window_size
        self.zoom = zoom
        self.camera_pos = np.array([0, 0], dtype=np.float32)
        self.dragging = False
        self.last_mouse_pos = np.array([0, 0])
        self.images = []
        self.draw_commands = []#叠加一些绘制效果
        self.draw_cache = None  # 缓存绘制的内容

        cv2.namedWindow('Image Viewer')
        cv2.setMouseCallback('Image Viewer', self.mouse_callback)



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
    def add_image(self, path, pos):
        self.images.append({'path': path, 'pos': pos, 'is_active': False, 'img': None})


    def draw_pixel(self, view, world_pos, color):
        view_pos = (world_pos - self.camera_pos) * self.zoom
        view_pos = view_pos.astype(int)
        if 0 <= view_pos[0] < self.window_size[0] and 0 <= view_pos[1] < self.window_size[1]:
            view[view_pos[1], view_pos[0]] = color
        

    def draw_line(self, view, start_pos, end_pos, color):
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        num_points = int(np.linalg.norm(end_pos - start_pos))
        for i in range(num_points + 1):
            t = i / num_points
            world_pos = (1 - t) * start_pos + t * end_pos
            self.draw_pixel(view, world_pos, color)

    def update_draw_cache(self):
        # 创建一个与窗口大小相同的黑色背景
        cache = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        for command in self.draw_commands:
            if command['type'] == 'line':
                self.draw_line(cache, command['start_pos'], command['end_pos'], command['color'])
            elif command['type'] == 'pixel':
                self.draw_pixel(cache, command['pos'], command['color'])
        self.draw_cache = cache

    def draw_others(self, view):
        if self.draw_cache is not None:
            # 重新计算缓存的绘制内容在当前视角下的位置
            cache = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            for command in self.draw_commands:
                if command['type'] == 'line':
                    self.draw_line(cache, command['start_pos'], command['end_pos'], command['color'])
                elif command['type'] == 'pixel':
                    self.draw_pixel(cache, command['pos'], command['color'])
            view[:] = cv2.addWeighted(view, 1, cache, 1, 0)



    def add_line(self, start_pos, end_pos, color):
        self.draw_commands.append({'type': 'line', 'start_pos': start_pos, 'end_pos': end_pos, 'color': color})
        self.update_draw_cache()

    def add_pixel(self, pos, color):  
        self.draw_commands.append({'type': 'pixel', 'pos': pos, 'color': color})
        self.update_draw_cache()

    def run(self):
        while True:
            # 创建黑色背景
            view = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

            # 计算相机视图范围
            top_left = self.camera_pos
            bottom_right = self.camera_pos + np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_top_left = top_left - np.array([self.window_size[0], self.window_size[1]]) / self.zoom
            extended_bottom_right = bottom_right + np.array([self.window_size[0], self.window_size[1]]) / self.zoom

            # 动态加载和卸载图像
            for image in self.images:
                pos = image['pos']
                img_top_left = pos[:2]
                img_bottom_right = pos[:2] + np.array([1000, 1000])  # 假设图像大小为1000x1000

                # 判断图像是否在视图范围内
                if (img_bottom_right[0] > extended_top_left[0] and img_top_left[0] < extended_bottom_right[0] and
                    img_bottom_right[1] > extended_top_left[1] and img_top_left[1] < extended_bottom_right[1]):
                    if not image['is_active']:
                        image['img'] = cv2.imread(image['path'])  # 加载图像
                        image['is_active'] = True
                        print('Load image:', image['path'])
                else:
                    if image['is_active']:
                        image['img'] = None  # 卸载图像
                        image['is_active'] = False

            # 渲染活动图像
            for image in self.images:
                if image['is_active']:
                    img = image['img']
                    pos = image['pos']
                    angle = pos[2]  # 获取旋转角度
                    img_top_left = pos[:2]
                    img_bottom_right = pos[:2] + np.array([img.shape[1], img.shape[0]])

                    # 判断图像是否在视图范围内
                    if (img_bottom_right[0] > top_left[0] and img_top_left[0] < bottom_right[0] and
                        img_bottom_right[1] > top_left[1] and img_top_left[1] < bottom_right[1]):
                        view_pos = (pos[:2] - top_left) * self.zoom
                        view_pos = view_pos.astype(int)

                        # 计算旋转中心和旋转矩阵
                        center = (img.shape[1] // 2, img.shape[0] // 2)
                        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img_rotated = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]))

                        # 缩放图像
                        img_resized = cv2.resize(img_rotated, (int(img.shape[1] * self.zoom), int(img.shape[0] * self.zoom)))

                        x1 = max(0, view_pos[0])
                        y1 = max(0, view_pos[1])
                        x2 = min(self.window_size[0], view_pos[0] + img_resized.shape[1])
                        y2 = min(self.window_size[1], view_pos[1] + img_resized.shape[0])

                        img_x1 = max(0, -view_pos[0])
                        img_y1 = max(0, -view_pos[1])
                        img_x2 = img_x1 + (x2 - x1)
                        img_y2 = img_y1 + (y2 - y1)

                        view[y1:y2, x1:x2] = img_resized[img_y1:img_y2, img_x1:img_x2]

            self.draw_others(view)

            # 添加相机位置文本
            cv2.putText(view, f'Camera Pos: {self.camera_pos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 添加相机位置文本
            cv2.putText(view, "press esc to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 显示图像
            cv2.imshow('Image Viewer', view)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
                break
        cv2.destroyAllWindows()



if __name__ == "__main__":

    viewer = ImageViewer()


    viewer.add_image("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0244.JPG", np.array([0, 0,0]))
    # viewer.add_image("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0053.JPG", np.array([-213.92524792,-808.49323945,-1.75539]))
    # viewer.add_image("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0053.JPG", np.array([-200,-790,-3]))
    viewer.add_image("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0245.JPG", np.array([-232.95030331, -857.97712893,0]))
    viewer.add_image("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0246.JPG", np.array([ -382.54631585, -1706.48444968,0]))
    # viewer.add_image("./03_0003.JPG", np.array([1000, -1000]))
    # viewer.add_image("./03_0004.JPG", np.array([-10000, -1000]))
    # viewer.add_line( [0, 0],[1000, 5000], (0, 255, 0))

    # ============这里是一个加载文件夹中所有图像的例子================
    # image_dir = 'D:/Dataset/UAV_VisLoc_dataset/03/drone/'
    # image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.JPG', '.jpeg', '.png'))]
    # images = []
    # for path in image_paths:
    #     # 随机生成图像位置，确保每个图像最小相距1000以上
    #     pos = np.array([random.randint(0, 100000), random.randint(0, 100000)])
    #     viewer.add_image(path,pos)
    
    
    
    viewer.run()



