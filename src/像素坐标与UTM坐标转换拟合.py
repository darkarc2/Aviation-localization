import pyproj
import numpy as np
from scipy.spatial.transform import Rotation as R




# 计算前三张图片再utm坐标系下的坐标
wgs84 = pyproj.Proj(init='epsg:4326')  # 定义WGS84坐标系
utm = pyproj.Proj(proj='utm', zone=50, ellps='WGS84') # 定义ENU坐标系,Zone


u1,v1=0,0
u2,v2=-207.47122192, -919.08453369
u3,v3=-303.48322296, -1682.33013916

u4,v4= -555.54845428, -2329.315979
u5,v5= -669.15367126, -3079.30212402
u6,v6= -769.40084839, -3971.13549805

# 经纬度坐标
lat1, lon1 =32.30042695,119.8866631
lat2, lon2 =32.30094834,119.8858724
lat3, lon3=32.30147546,119.8850817

lat4, lon4=32.30383604,119.88978
lat5, lon5=32.30436317,119.8889893
lat6, lon6=32.30487883,119.8881757


# 将经纬度坐标转换为UTM坐标
x1, y1 = pyproj.transform(wgs84, utm, lon1, lat1)
x2, y2 = pyproj.transform(wgs84, utm, lon2, lat2)
x3, y3 = pyproj.transform(wgs84, utm, lon3, lat3)
x4, y4 = pyproj.transform(wgs84, utm, lon4, lat4)
x5, y5 = pyproj.transform(wgs84, utm, lon5, lat5)
x6, y6 = pyproj.transform(wgs84, utm, lon6, lat6)


# 计算两个点之间的平面距离
distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

print(f"点1的UTM坐标: ({x1}, {y1})")
print(f"点2的UTM坐标: ({x2}, {y2})")
# print(f"点3的UTM坐标: ({x3}, {y3})")
# 计算ENU坐标
dx = x2 - x1
dy = y2 - y1
print(f"dx: {dx:.2f} 米, dy: {dy:.2f} 米")
print(f"两个点之间的平面距离: {distance:.2f} 米")






import numpy as np

# 已知点
p1 = (x1, y1)
p2 = (x2, y2)
p3 = (x3, y3)
p4 = (x4, y4)
p5 = (x5, y5)
p6 = (x6, y6)

# 原始坐标 (u, v)
# u = np.array([0, u2, u3, u4, u5, u6])
# v = np.array([0, v2,v3,v4,v5,v6])
u = np.array([0, u2])
v = np.array([0, v2])

# 目标坐标 (p)
p = np.array([
    [p1[0], p1[1]],
    [p2[0], p2[1]],
    # [p3[0], p3[1]],
    # [p4[0], p4[1]],
    # [p5[0], p5[1]],
    # [p6[0], p6[1]]
])

# 构建矩阵 A 和向量 B
A = []
B = []

for i in range(len(u)):
    A.append([u[i], v[i], 1, 0, 0, 0])
    A.append([0, 0, 0, u[i], v[i], 1])
    B.append(p[i][0])
    B.append(p[i][1])

A = np.array(A)
B = np.array(B)

# 使用最小二乘法求解
params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

# 提取参数
a, b, e, c, d, f = params

# 打印结果
print(f"变换矩阵: \n[{a}, {b},{c}, {d}]")
print(f"平移向量: [{e}, {f}]")

# 验证函数
def transform_point(u, v):
    a,b,c,d=-0.16923672929767528, 0.12093727164720008,0.1324161747470498, -0.09062548226614338
    e,f=771817.6505573406, 3577397.3732869313
    x = a * u + b * v + e
    y = c * u + d * v + f
    return x, y

# 验证 p1, p2, p3
for i in range(len(u)):
    x, y = transform_point(u[i], v[i])
    print(f"原始坐标: ({u[i]}, {v[i]}) -> 计算坐标: ({x}, {y}) -> 目标坐标: {p[i]}")

# 验证 p4
x, y = transform_point(u4, v4)
print(f"原始坐标: ({u4}, {v4}) -> 计算坐标: ({x}, {y}) -> 目标坐标: {p4}")
