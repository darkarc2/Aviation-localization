import pyproj
import numpy as np
from scipy.spatial.transform import Rotation as R

# 计算前三张图片再utm坐标系下的坐标
wgs84 = pyproj.Proj(init='epsg:4326')  # 定义WGS84坐标系
utm = pyproj.Proj(proj='utm', zone=50, ellps='WGS84') # 定义ENU坐标系,Zone

# 经纬度坐标
lat1, lon1 = 32.309694723574005, 119.88175977125
lat2, lon2 = 32.30957135,119.8809793
lat3, lon3=32.32883992,119.851346

# 将经纬度坐标转换为UTM坐标
x1, y1 = pyproj.transform(wgs84, utm, lon1, lat1)
x2, y2 = pyproj.transform(wgs84, utm, lon2, lat2)
x3, y3 = pyproj.transform(wgs84, utm, lon3, lat3)



# 计算两个点之间的平面距离
distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

print(f"点1的UTM坐标: ({x1}, {y1})")
print(f"点2的UTM坐标: ({x2}, {y2})")
print(f"点3的UTM坐标: ({x3}, {y3})")
# 计算ENU坐标
dx = x2 - x1
dy = y2 - y1
print(f"dx: {dx:.2f} 米, dy: {dy:.2f} 米")
print(f"两个点之间的平面距离: {distance:.2f} 米")

u,v=-232.95030331, -857.97712893
dis_uv=(u**2+v**2)**0.5

print(f"像素与实际距离比例ku为{dx/u}")
print(f"像素与实际距离比例kv为{dy/v}")
k=distance/dis_uv
print(f"像素与实际距离比例k为{distance/dis_uv}")
P2=[u*k,v*k]
P1=[dx,dy]
# P1=R@P2
print(f"像素坐标为{P2}")
def calculate_rotation_matrix(P1, P2):
    # 将点转换为向量
    v1 = np.array(P1)
    v2 = np.array(P2)

    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 归一化向量
    v1_normalized = v1 / norm_v1
    v2_normalized = v2 / norm_v2

    # 计算向量之间的夹角
    cos_theta = np.dot(v1_normalized, v2_normalized)
    sin_theta = np.cross(v1_normalized, v2_normalized)

    # 构建旋转矩阵
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    return R
R = calculate_rotation_matrix(P2, P1)
print(f"旋转矩阵为:\n{R}")
print(f"验证为:\n{R@P2}")
print(f"为:\n{P1}")

u1,v1=0,0
u2,v2= -232.95030331, -857.97712893
u3,v3= -382.54631585, -1706.48444968
u4,v4= -548.98004176, -2521.3039934 


p1=(768562.1758137611, 3580349.159205035)
p2=(768485.1368453965, 3580403.70158688)
p3=(768406.9819545087, 3580459.4866603795)
p4=(768330.9843558057, 3580515.3297081315)

import numpy as np

# 已知点
p1 = (768562.1758137611, 3580349.159205035)
p2 = (768485.1368453965, 3580403.70158688)
p3 = (768406.9819545087, 3580459.4866603795)

# 原始坐标 (u, v)
u = np.array([0, -232.95030331, -382.54631585])
v = np.array([0, -857.97712893, -1706.48444968])

# 目标坐标 (p)
p = np.array([
    [768562.1758137611, 3580349.159205035],
    [768485.1368453965, 3580403.70158688],
    [768406.9819545087, 3580459.4866603795]
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
print(f"变换矩阵: \n[{a}, {b}，{c}, {d}]")
print(f"平移向量: [{e}, {f}]")

# 验证函数
def transform_point(u, v):
    a,b,c,d=-0.024339612852506524, 0.09639987552971423,0.02283516200259328, -0.06977090383329672
    e,f=768562.1758137619, 3580349.1592050353
    x = a * u + b * v + e
    y = c * u + d * v + f
    return x, y

# 验证 p1, p2, p3
for i in range(len(u)):
    x, y = transform_point(u[i], v[i], a, b, c, d, e, f)
    print(f"原始坐标: ({u[i]}, {v[i]}) -> 计算坐标: ({x}, {y}) -> 目标坐标: {p[i]}")

x, y = transform_point(u4, v4, a, b, c, d, e, f)
print(f"原始坐标: ({u4}, {v4}) -> 计算坐标: ({x}, {y}) -> 目标坐标: {p4}")