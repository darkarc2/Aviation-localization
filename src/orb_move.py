import cv2
import numpy as np
import os

# 构建图像金字塔
def build_pyramid(src, levels, scale):
    pyramid = [src]
    for i in range(1, levels):
        dst = cv2.resize(pyramid[i - 1], (0, 0), fx=scale, fy=scale)  # 缩放图像
        pyramid.append(dst)
    return pyramid

# 计算图像之间的位移
def compute_displacement(img_1, img_2, pyramid_levels, scale):
    pyramid_1 = build_pyramid(img_1, pyramid_levels, scale)
    pyramid_2 = build_pyramid(img_2, pyramid_levels, scale)

    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    final_matches = []
    keypoints_1, keypoints_2 = [], []

    for i in range(pyramid_levels - 1, -1, -1):
        kp1, des1 = detector.detectAndCompute(pyramid_1[i], None)
        kp2, des2 = detector.detectAndCompute(pyramid_2[i], None)

        matches = matcher.match(des1, des2)

        min_dist = min(matches, key=lambda x: x.distance).distance
        max_dist = max(matches, key=lambda x: x.distance).distance

        good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 30.0)]

        if i == 0:
            final_matches = good_matches
            keypoints_1 = kp1
            keypoints_2 = kp2

    # 计算匹配点连线的方向
    directions = []
    for match in final_matches:
        pt1 = np.array(kp1[match.queryIdx].pt)
        pt2 = np.array(kp2[match.trainIdx].pt)
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
    filtered_matches = []
    for match, direction in zip(final_matches, directions):
        if abs(direction - main_direction) <= threshold:
            filtered_matches.append(match)

    displacement = np.zeros(2)
    if filtered_matches:
        for match in filtered_matches:
            displacement += np.array(kp2[match.trainIdx].pt) - np.array(kp1[match.queryIdx].pt)
        displacement /= len(filtered_matches)

    return displacement, filtered_matches, keypoints_1, keypoints_2

# 鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    global dragging, last_x, last_y, offset_x, offset_y, zoom_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        last_x, last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx, dy = x - last_x, y - last_y
            offset_x += dx
            offset_y += dy
            last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_factor *= 1.1
        else:
            zoom_factor /= 1.1

def main(img_directory):
    global dragging, last_x, last_y, offset_x, offset_y, zoom_factor

    # 初始化全局变量
    dragging = False
    last_x, last_y = 0, 0
    offset_x, offset_y = 0, 0
    zoom_factor = 1.0

    # 获取目录中的所有图片文件
    img_files = [os.path.join(img_directory, f) for f in os.listdir(img_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    img_files.sort()

    if len(img_files) < 2:
        print("Not enough images in the directory.")
        return

    pyramid_levels = 5  # 金字塔层数
    scale = 0.5  # 缩放倍数
    scale_factor = 0.01  # 缩放因子，用于调整位移的大小
    displacements = []
    current_position = np.zeros(2)

    cv2.namedWindow("Trajectory", cv2.WINDOW_AUTOSIZE)  # 创建显示轨迹的窗口
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)  # 创建显示图片的窗口
    cv2.namedWindow("Global Map", cv2.WINDOW_AUTOSIZE)  # 创建显示全局地图的窗口
    cv2.setMouseCallback("Global Map", mouse_callback)  # 设置鼠标回调函数

    traj = np.zeros((600, 600, 3), dtype=np.uint8)  # 初始化轨迹图像

    map_width, map_height = 5000, 5000  # 根据需要调整大小
    global_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)  # 初始化全局地图
    map_offset = np.array([map_width // 2, map_height // 2])  # 地图偏移量

    for i in range(1, len(img_files)):
        img_1 = cv2.imread(img_files[i - 1])
        img_2 = cv2.imread(img_files[i])

        if img_1 is None or img_2 is None:
            print("Could not open or find the image!")
            return

        displacement, final_matches, keypoints_1, keypoints_2 = compute_displacement(img_1, img_2, pyramid_levels, scale)
        current_position += displacement  # 更新当前位置
        displacements.append(current_position)

        # 绘制轨迹
        draw_position = current_position * scale_factor + np.array([traj.shape[1] // 2, traj.shape[0] // 2])
        cv2.circle(traj, tuple(draw_position.astype(int)), 1, (0, 0, 255), 2)
        cv2.imshow("Trajectory", traj)

        points1 = np.float32([kp.pt for kp in keypoints_1])
        points2 = np.float32([kp.pt for kp in keypoints_2])

        if len(points1) >= 4 and len(points2) >= 4:
            H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995)
            
            # 获取原始图像的高宽
            h1, w1 = img_1.shape[:2]
            h2, w2 = img_2.shape[:2]
            
            # 获取两幅图的边界坐标
            img1_pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
            img2_pts = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
            
            # 获取 img1 的边界坐标变换之后的坐标
            img1_transform = cv2.perspectiveTransform(img1_pts, H)
            
            # 把 img2 和转换后的边界坐标连接起来
            result_pts = np.concatenate((img2_pts, img1_transform), axis=0)
            
            # 获取拼接图像的边界
            [x_min, y_min] = np.int32(result_pts.min(axis=0).ravel() - 1)
            [x_max, y_max] = np.int32(result_pts.max(axis=0).ravel() + 1)
            
            # 手动构造平移矩阵
            M = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            
            # 对 img1 进行平移和透视操作
            result = cv2.warpPerspective(img_1, M.dot(H), (x_max - x_min, y_max - y_min))
            
            # 显示拼接结果
            cv2.imshow("Stitched Image", result)
            cv2.waitKey(0)

            # 确保 result 图像能够正确地放置在 global_map 中
            x_offset = map_offset[0] + x_min
            y_offset = map_offset[1] + y_min
            
            # 计算 result 图像在 global_map 中的目标区域
            y_end = y_offset + result.shape[0]
            x_end = x_offset + result.shape[1]
            
            # 确保目标区域在 global_map 的边界内
            if y_end > global_map.shape[0]:
                result = result[:global_map.shape[0] - y_offset, :]
                y_end = global_map.shape[0]
            
            if x_end > global_map.shape[1]:
                result = result[:, :global_map.shape[1] - x_offset]
                x_end = global_map.shape[1]
            
            # 将 result 图像叠加到 global_map 上
            global_map[y_offset:y_end, x_offset:x_end] = result
            
            # 在全局地图上显示当前位置
            map_position = current_position * scale_factor + map_offset
            cv2.circle(global_map, tuple(map_position.astype(int)), 5, (0, 0, 255), -1)

        # 显示匹配特征点的图片
        img_matches = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, final_matches, None)
        img_small = cv2.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Image", img_small)

        cv2.waitKey(100)  # 等待100毫秒

    while True:
        # 计算当前视图范围
        view_width = int(global_map.shape[1] / zoom_factor)
        view_height = int(global_map.shape[0] / zoom_factor)
        x_start = max(0, offset_x)
        y_start = max(0, offset_y)
        x_end = min(global_map.shape[1], x_start + view_width)
        y_end = min(global_map.shape[0], y_start + view_height)

        # 打印调试信息
        print(f"View: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")

        # 提取当前视图范围内的图像块
        view = global_map[y_start:y_end, x_start:x_end]
        view = cv2.resize(view, (global_map.shape[1], global_map.shape[0]))

        cv2.imshow("Global Map", view)

        key = cv2.waitKey(30)
        if key == 27:  # 按下ESC键退出
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/")