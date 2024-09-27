import cv2
import numpy as np

# 读取图片
img1 = cv2.imread("/home/arc/works/review_prj/UAV_slam/src/03_0001.JPG")
img2 = cv2.imread("/home/arc/works/review_prj/UAV_slam/src/03_0002.JPG")

# 构建5层金字塔
pyramid1 = [img1]
pyramid2 = [img2]
for i in range(4):
    pyramid1.append(cv2.pyrDown(pyramid1[-1]))
    pyramid2.append(cv2.pyrDown(pyramid2[-1]))

# 初始化光流点
p0 = None

# 逐层计算光流并优化
for i in range(4, -1, -1):
    gray1 = cv2.cvtColor(pyramid1[i], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(pyramid2[i], cv2.COLOR_BGR2GRAY)

    if p0 is None:
        # 在最小层检测角点
        p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    else:
        # 将上一层的光流点放大到当前层
        p0 = p0 * 2

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, winSize=(15, 15), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 选择好的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制光流结果
    mask = np.zeros_like(pyramid2[i])
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        pyramid2[i] = cv2.circle(pyramid2[i], (int(a), int(b)), 5, (0, 255, 0), -1)

    img = cv2.add(pyramid2[i], mask)


    # 展示结果
    cv2.imshow('Optical Flow', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()