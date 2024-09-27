import cv2
import numpy as np
import os
import time
from show_img import ImageViewer






def compute_translation(points1,points2):
	
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

    # 保存平移向量到pose变量
    deta_pose = translation_vector
    return deta_pose





# 读取图像
image_path1= "/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG"
img1 = cv2.imread(image_path1,0)
# 读取图像
image_path2 = "/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG"
img2 = cv2.imread(image_path2,0)


# 创建ORB检测器
orb = cv2.ORB_create()

# 检测关键点和描述符
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


#===========================原始orb匹配方法===========================   
start = time.time()
# # 使用BFMatcher进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

end=time.time()
print("spend_time:",end-start)
# 获取匹配点
points1 = [keypoints1[m.queryIdx].pt for m in matches]
points2 = [keypoints2[m.trainIdx].pt for m in matches]

# 计算得到图像某个位置上，4x4像素块，光度均值
def get_img_value(img, p):
    x = int(p.pt[0])
    y = int(p.pt[1])
    # 提取4x4像素块
    block = img[y-2:y+2, x-2:x+2]
    # 计算光度均值
    mean_value = np.mean(block)
    return mean_value



#====================引力匹配方法====================
I1=[]
for i in range(len(keypoints1)):
    I1.append(img1[int(keypoints1[i].pt[1]),int(keypoints1[i].pt[0])])
F=[]

# 初始化合力数组
forces = []
start = time.time()
for i in range(len(keypoints1)):
    F2=[]
    p1=keypoints1[i]
    I1=get_img_value(img1,p1)

    force_vector = np.array([0.0, 0.0])

    for j in range(len(keypoints2)):
        p2=keypoints2[j]
        # I2=img2[int(p2.pt[1]),int(p2.pt[0])]
        I2=get_img_value(img1,p2)
        
        dist=np.array([p2.pt[1]-p1.pt[1],p2.pt[0]-p1.pt[0]])
        dist=dist.transpose()@dist#距离，越小越好

        dI=abs(I2-I1)#光度差，越小越好

        feature=abs((I1-128)*(I2-128))#特征，越亮或者越暗，说明特征明显，越大越好

        f=feature/(100+dist/1000)/(1+dI)*1000#引力f
        # f=feature/(1+dI)#引力f
        dist_vector = np.array([p2.pt[1] - p1.pt[1], p2.pt[0] - p1.pt[0]])
         # 计算吸引力矢量
        force_vector += f * (dist_vector / dist)
        F2.append(f)
    forces.append(force_vector)
    F.append(F2)
end=time.time()
print("spend_time:",end-start)

mean=np.mean(F)
# for i in F:
print(mean)

# 计算总的图像合力
total_force = np.sum(forces, axis=0)/10





# import cv2
# import numpy as np

def add_image(canvas, image_path, position):
    image = cv2.imread(image_path)
    x, y = position
    h, w, _ = image.shape
    canvas[y:y+h, x:x+w] = image

def add_line(canvas, p1, p2, color):
    cv2.line(canvas, tuple(p1), tuple(p2), color, 2)

# 创建一个足够大的画布
canvas_height = 4000
canvas_width = 8000
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# 添加图像
add_image(canvas, "/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG", (0, 0))
add_image(canvas, "/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG", (4000, 0))


# 绘制关键点收到的力的线条
for i in range(1):
    for j in range(len(keypoints2)):
        if F[i][j]:
            p1 = [int(keypoints1[i].pt[0]), int(keypoints1[i].pt[1])]
            p2 = [int(keypoints2[j].pt[0] + 4000), int(keypoints2[j].pt[1])]
            max_f=np.max(F[i])
            add_line(canvas, p1, p2, (0, int(F[i][j]/max_f*255), int(F[i][j]/max_f*255)))
    print("add:", i)
# 绘制关键点的合力线条
for i, force in enumerate(forces):
    p1 = (int(keypoints1[i].pt[0]), int(keypoints1[i].pt[1]))
    p2 = (int(p1[0] + force[0]), int(p1[1] + force[1]))
    cv2.arrowedLine(canvas, p1, p2, (0, 255, 0), 2)

# 绘制图像的总合力线条
image_center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
total_force_end = (int(image_center[0] + total_force[0]), int(image_center[1] + total_force[1]))
cv2.arrowedLine(canvas, image_center, total_force_end, (255, 0, 0), 2)

# 显示结果
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
# add_image("/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG", np.array([0, 0]))
# add_image("/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG", np.array([4000, 0]))
# # viewer.add_image("./imgs/03_0003.JPG", np.array([1000, -1000]))
# # viewer.add_image("./imgs/03_0004.JPG", np.array([-10000, -1000]))

# for i in range(10):
#     # viewer.add_pixel(keypoints1[i].pt, (255, 0, 0))
#     for j in range(len(keypoints2)):
#         if F[i][j]:
#             p1=[keypoints1[i].pt[0],keypoints1[i].pt[1]]
#             p2=[keypoints2[j].pt[0]+4000,keypoints2[j].pt[1]]
#             add_line(p1, p2, (0, 255, 0))
#     print("add:",i)

# p1=[keypoints1[0].pt[0],keypoints1[0].pt[1]]
# p2=[keypoints2[0].pt[0]+4000,keypoints2[0].pt[1]]
# import random
# for i in range(1000):
#     p1=[random.randint(0, 1000),random.randint(0, 1000)]
#     p2=[random.randint(0, 1000),random.randint(0, 1000)]
#     viewer.add_line(p1, p2, (0, 255, 0))
# viewer.add_line(p1, p2, (0, 255, 0))
# viewer.add_line(p1, p2, (0, 255, 0))
# viewer.add_line(p1, p2, (0, 255, 0))
# viewer.add_line(p1, p2, (0, 255, 0))

