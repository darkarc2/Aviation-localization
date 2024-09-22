import cv2
import numpy as np

# 加载图片路径
image_paths = ['src/03_0001.JPG', 'src/03_0002.JPG']

# 初始化相机参数
camera_pos = np.array([0, 0], dtype=np.float32)  # 相机位置
zoom = 1.0  # 缩放比例

# 定义窗口大小
window_size = (800, 600)  # 窗口宽度和高度

# 初始化图像信息
images = [
    {'path': 'src/03_0001.JPG', 'pos': np.array([0, 0]), 'is_active': False, 'img': None},
    {'path': 'src/03_0002.JPG', 'pos': np.array([10000, 10000]), 'is_active': False, 'img': None}
]

# 鼠标回调函数，用于处理鼠标事件
def mouse_callback(event, x, y, flags, param):
    global camera_pos, zoom, dragging, last_mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下事件
        dragging = True
        last_mouse_pos = np.array([x, y])
    elif event == cv2.EVENT_MOUSEMOVE and dragging:  # 鼠标移动事件
        delta = (np.array([x, y]) - last_mouse_pos) / zoom
        camera_pos -= delta
        last_mouse_pos = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP:  # 左键抬起事件
        dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:  # 鼠标滚轮事件
        if flags > 0:
            zoom *= 1.1
        else:
            zoom /= 1.1

# 初始化拖动状态
dragging = False
last_mouse_pos = np.array([0, 0])

# 创建窗口并设置鼠标回调
cv2.namedWindow('Image Viewer')
cv2.setMouseCallback('Image Viewer', mouse_callback)

while True:
    # 创建黑色背景
    view = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

    # 计算相机视图范围
    top_left = camera_pos
    bottom_right = camera_pos + np.array([window_size[0], window_size[1]]) / zoom
    extended_top_left = top_left - np.array([window_size[0], window_size[1]]) / zoom
    extended_bottom_right = bottom_right + np.array([window_size[0], window_size[1]]) / zoom

    # 动态加载和卸载图像
    for image in images:
        pos = image['pos']
        img_top_left = pos
        img_bottom_right = pos + np.array([1000, 1000])  # 假设图像大小为1000x1000

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
    for image in images:
        if image['is_active']:
            img = image['img']
            pos = image['pos']
            img_top_left = pos
            img_bottom_right = pos + np.array([img.shape[1], img.shape[0]])

            # 判断图像是否在视图范围内
            if (img_bottom_right[0] > top_left[0] and img_top_left[0] < bottom_right[0] and
                img_bottom_right[1] > top_left[1] and img_top_left[1] < bottom_right[1]):
                view_pos = (pos - top_left) * zoom
                view_pos = view_pos.astype(int)

                x1 = max(0, view_pos[0])
                y1 = max(0, view_pos[1])
                x2 = min(window_size[0], view_pos[0] + int(img.shape[1] * zoom))
                y2 = min(window_size[1], view_pos[1] + int(img.shape[0] * zoom))

                img_x1 = max(0, -view_pos[0])
                img_y1 = max(0, -view_pos[1])
                img_x2 = img_x1 + (x2 - x1)
                img_y2 = img_y1 + (y2 - y1)

                img_resized = cv2.resize(img, (int(img.shape[1] * zoom), int(img.shape[0] * zoom)))

                view[y1:y2, x1:x2] = img_resized[img_y1:img_y2, img_x1:img_x2]

    # 添加相机位置文本
    cv2.putText(view, f'Camera Pos: {camera_pos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 显示图像
    cv2.imshow('Image Viewer', view)

    # 退出条件
    if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
        break

cv2.destroyAllWindows()