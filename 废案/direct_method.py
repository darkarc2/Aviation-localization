import numpy as np
import cv2
import sophuspy as sp



# 线性插值获得图片某点的value
def get_pixel_value(img, x, y):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x >= img.shape[1]:
        x = img.shape[1] - 1
    if y >= img.shape[0]:
        y = img.shape[0] - 1
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, img.shape[1] - 1)
    y1 = min(y0 + 1, img.shape[0] - 1)
    a = x - x0
    b = y - y0
    return (1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x1] + (1 - a) * b * img[y1, x0] + a * b * img[y1, x1]

class JacobianAccumulator:
    def __init__(self, img1, img2, px_ref, depth_ref):
        self.img1 = img1
        self.img2 = img2
        self.px_ref = px_ref
        self.depth_ref = depth_ref
        self.projection = np.zeros((len(px_ref), 2))
        self.H = np.zeros((6, 6))
        self.b = np.zeros(6)
        self.cost = 0

    def accumulate_jacobian(self, range_start, range_end,T21):
        half_patch_size = 1
        cnt_good = 0
        hessian = np.zeros((6, 6))
        bias = np.zeros(6)
        cost_tmp = 0

        global fx, fy, cx, cy
        for i in range(range_start, range_end):
            point_ref = self.depth_ref[i] * np.array([(self.px_ref[i][0] - cx) / fx, (self.px_ref[i][1] - cy) / fy, 1])
            point_cur = T21 * point_ref
            if point_cur[2] < 0:
                continue

            u = fx * point_cur[0] / point_cur[2] + cx
            v = fy * point_cur[1] / point_cur[2] + cy
            if u < half_patch_size or u > self.img2.shape[1] - half_patch_size or v < half_patch_size or v > self.img2.shape[0] - half_patch_size:
                continue

            self.projection[i] = [u, v]
            X, Y, Z = point_cur
            Z2 = Z * Z
            Z_inv = 1.0 / Z
            Z2_inv = Z_inv * Z_inv
            cnt_good += 1

            for x in range(-half_patch_size, half_patch_size + 1):
                for y in range(-half_patch_size, half_patch_size + 1):
                    error = get_pixel_value(self.img1, self.px_ref[i][0] + x, self.px_ref[i][1] + y) - get_pixel_value(self.img2, u + x, v + y)
                    J_pixel_xi = np.zeros((2, 6))
                    J_img_pixel = np.zeros(2)

                    J_pixel_xi[0, 0] = fx * Z_inv
                    J_pixel_xi[0, 1] = 0
                    J_pixel_xi[0, 2] = -fx * X * Z2_inv
                    J_pixel_xi[0, 3] = -fx * X * Y * Z2_inv
                    J_pixel_xi[0, 4] = fx + fx * X * X * Z2_inv
                    J_pixel_xi[0, 5] = -fx * Y * Z_inv

                    J_pixel_xi[1, 0] = 0
                    J_pixel_xi[1, 1] = fy * Z_inv
                    J_pixel_xi[1, 2] = -fy * Y * Z2_inv
                    J_pixel_xi[1, 3] = -fy - fy * Y * Y * Z2_inv
                    J_pixel_xi[1, 4] = fy * X * Y * Z2_inv
                    J_pixel_xi[1, 5] = fy * X * Z_inv

                    J_img_pixel[0] = 0.5 * (get_pixel_value(self.img2, u + 1 + x, v + y) - get_pixel_value(self.img2, u - 1 + x, v + y))
                    J_img_pixel[1] = 0.5 * (get_pixel_value(self.img2, u + x, v + 1 + y) - get_pixel_value(self.img2, u + x, v - 1 + y))

                    J = -1.0 * (J_img_pixel @ J_pixel_xi)

                    hessian += np.outer(J, J)
                    bias += -error * J
                    cost_tmp += error * error

        if cnt_good:
            self.H += hessian
            self.b += bias
            self.cost += cost_tmp / cnt_good

def direct_pose_estimation_single_layer(img1, img2, px_ref, depth_ref, T21):
    iterations = 10
    jaco_accu = JacobianAccumulator(img1, img2, px_ref, depth_ref)

    for iter in range(iterations):
        jaco_accu.H = np.zeros((6, 6))
        jaco_accu.b = np.zeros(6)
        jaco_accu.cost = 0
        jaco_accu.accumulate_jacobian(0, len(px_ref), T21[0])

        H = jaco_accu.H
        b = jaco_accu.b

        update = np.linalg.solve(H, b)

        T21[0] = sp.SE3.exp(update) * T21[0]
        cost = jaco_accu.cost

        if np.isnan(update[0]):
            print("update is nan")
            break
        if iter > 0 and cost > lastCost:
            print(f"cost increased: {cost}, {lastCost}")
            break
        if np.linalg.norm(update) < 1e-3:
            break

        lastCost = cost
        print(f"iteration: {iter}, cost: {cost}")

    print("T21 = \n", T21[0].matrix())
    # 将灰度图像转换为彩色图像
    img2_show = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 获取投影点
    projection = jaco_accu.projection

    # 初始化总差值
    total_diff_x = 0
    total_diff_y = 0
    count = 0

    # 绘制圆和线
    for i in range(len(px_ref)):
        p_ref = px_ref[i]
        p_cur = projection[i]
        if p_cur[0] > 0 and p_cur[1] > 0:
            cv2.circle(img2_show, (int(p_cur[0]), int(p_cur[1])), 2, (0, 250, 0), 2)
            cv2.line(img2_show, (int(p_ref[0]), int(p_ref[1])), (int(p_cur[0]), int(p_cur[1])), (0, 250, 0))
            
            # 计算差值并累加
            diff_x = p_cur[0] - p_ref[0]
            diff_y = p_cur[1] - p_ref[1]
            total_diff_x += diff_x
            total_diff_y += diff_y
            count += 1

    # 计算平均差值
    if count > 0:
        avg_diff_x = total_diff_x / count
        avg_diff_y = total_diff_y / count
        print(f"平均像素坐标差 - x轴: {avg_diff_x}, y轴: {avg_diff_y}")
    else:
        print("没有有效的点进行计算")

    # 显示图像
    cv2.imshow("current", img2_show)
    cv2.waitKey(0)

    
def direct_pose_estimation_multi_layer(img1, img2, px_ref, depth_ref, T21):
    # parameters
    pyramids = 5
    pyramid_scale = 0.5
    scales = [1.0, 0.5, 0.25, 0.125, 0.0625]

    # create pyramids
    pyr1 = [img1]
    pyr2 = [img2]
    for i in range(1, pyramids):
        pyr1.append(cv2.resize(pyr1[i - 1], (0, 0), fx=pyramid_scale, fy=pyramid_scale))
        pyr2.append(cv2.resize(pyr2[i - 1], (0, 0), fx=pyramid_scale, fy=pyramid_scale))

    # backup the old values
    global fx, fy, cx, cy
    fxG, fyG, cxG, cyG = fx, fy, cx, cy

    for level in range(pyramids - 1, -1, -1):
        px_ref_pyr = [scales[level] * np.array(px) for px in px_ref]

        # scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level]
        fy = fyG * scales[level]
        cx = cxG * scales[level]
        cy = cyG * scales[level]

        direct_pose_estimation_single_layer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21)

    # restore the old values
    fx, fy, cx, cy = fxG, fyG, cxG, cyG
    


# Camera intrinsics
# fx = 7752.65
# fy = 7735
# cx = 3976
# cy = 2652
fx = 77.5265
fy = 77.35
cx = 39.76
cy = 26.52
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
# baseline
# baseline = 0.573
# paths
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
# start_file = "/home/arc/works/Gao_book/slambook2/ch8/left.png"
# img2_path = "/home/arc/works/Gao_book/slambook2/ch8/000002.png"
# disparity_file = "/home/arc/works/Gao_book/slambook2/ch8/disparity.png"
start_file="/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG"
img2_path="/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG"


def main():
    left_img = cv2.imread(start_file, 0)
    # disparity_img = cv2.imread(disparity_file, 0)

    pixels_ref = []
    depth_ref = []

    for x in range(20, left_img.shape[1] - 20, 100):
        for y in range(20, left_img.shape[0] - 20, 100):
            pixels_ref.append([x, y])
            # disparity = disparity_img[y, x]
            # depth = fx * baseline / disparity
            depth_ref.append(1)
        # 创建ORB检测器
    # orb = cv2.ORB_create()

    # # 检测关键点和描述符
    # keypoints1, descriptors1 = orb.detectAndCompute(left_img, None)
    # for i in range(len(keypoints1)):
    #     pixels_ref.append(keypoints1[i].pt)
    #     depth_ref.append(466)
    

    # # 设置初始位移
    initial_translation = np.array([0,0, 0]).reshape(3, 1)
    initial_rotation = np.eye(3)  # 使用单位矩阵作为初始旋转矩阵

    # 创建 SE3 对象
    T21 = sp.SE3(initial_rotation, initial_translation)  
    # T21 = sp.SE3()  

    T21=[T21]
    img = cv2.imread(img2_path, 0)
    # direct_pose_estimation_single_layer(left_img, img, pixels_ref, depth_ref, T21)
    direct_pose_estimation_multi_layer(left_img, img, pixels_ref, depth_ref, T21)

    T21 = np.array(T21[0].matrix())
    p0=np.array([0,0,466,1])
    p1=T21@p0
    uv_pose1=1/466*K@p1[:3]
    print(uv_pose1)

if __name__ == "__main__":
    main()