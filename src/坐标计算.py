
import pyproj
import numpy as np
from scipy.spatial.transform import Rotation as R

# 计算前三张图片再utm坐标系下的坐标
wgs84 = pyproj.Proj(init='epsg:4326')  # 定义WGS84坐标系
utm = pyproj.Proj(proj='utm', zone=50, ellps='WGS84') # 定义ENU坐标系,Zone

# 经纬度坐标
lat1, lon1 = 32.30462673, 119.8968847
lat2, lon2 = 32.30513666, 119.8960711
lat3, lon3=32.30566378,119.8952804

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





# pos=np.array([0,0,466])
# print(f"像素坐标为{1/466*K@pos}")




uv_pose1=np.array([3976,2652,1]) #图1的中心点
uv_pose2=np.array([-171.25253296,-886.85534668,0])+uv_pose1  #uv_pose2是uv_pose1的相对坐标
uv_pose3=np.array([ -238.22377014,-1665.41070557,0])+uv_pose1 #uv_pose3是uv_pose1的相对坐标

#相机模型测试

a=7952/35.9
b=5304/24
f=35

K = np.array([
    [f*a, 0, 3976],
    [0, f*b, 2652],
    [0, 0, 1]
])

# 相机坐标系下各图片中心点
k_inv=np.linalg.inv(K)
pose1_in_camera=466*k_inv@(uv_pose1)
pose2_in_camera=466*k_inv@(uv_pose2)
pose3_in_camera=466*k_inv@(uv_pose3)
print("=========相机坐标系下各图片中心点::=============")
print(f"pose1_in_camera为{pose1_in_camera}")
print(f"pose2_in_camera为{pose2_in_camera}")
print(f"pose3_in_camera为{pose3_in_camera}")
print("=========下面开始求解R，t::=============")


# 把第一张图片的中心点转到世界坐标系下，假设其在相机坐标系下的坐标为(0,0,446)
# Pwc=R@pc+t  ，@表示矩阵乘法
# 则很容易得到t=(x1,y1,0)
t=np.array([x1,y1,0])

#接下来要求R
# Pwc-t=R@pc
#这里就用第二张图片来求解R
# np.array(x2, y2,446)-t =R@pose1_in_camera

# 定义两个点的坐标
p = np.array([x2, y2,466])-t
print(f"Pwc-t为{p}")
# Pwc-t为[-78.16504048  54.48358478 466.        ]

# 计算旋转矩阵
rotation, _ = R.align_vectors([p], [pose2_in_camera])

# 打印旋转矩阵
print("旋转矩阵为：")
print(rotation.as_matrix())
print("t为：")
print(t)
#验证
print("验证结果为：")
print("Pwc-t为",rotation.as_matrix()@pose2_in_camera)



R=np.array([[-0.48546517,-0.83104347,-0.27145961],
 [ 0.85030412,-0.52101022, 0.07437244],
 [-0.20323996,-0.19471799, 0.959572  ]])
t=np.array([ 772767.90754028,3577889.15665317, 0])
#验证
print("图三utm坐标为，真值",x3,y3,466)
print("图三utm坐标为，计算",R@pose3_in_camera+t)
# print("图1utm坐标为，真值",x1,y1,466)
# print("图1utm坐标为，计算",R@pose1_in_camera+t)


# 定义一个函数将UTM坐标转换回经纬度坐标
def utm_to_latlon(x, y, zone=50):
	# 定义WGS84坐标系
    wgs84 = pyproj.CRS('EPSG:4326')
    # 定义UTM坐标系
    utm = pyproj.CRS(f'EPSG:326{zone}')
    # 创建转换器
    transformer = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)
    # 进行坐标转换
    lon, lat = transformer.transform(x, y)
    return lat, lon


print("图三的经纬度坐标为，真值",utm_to_latlon(x3, y3))
p3=R@pose3_in_camera+t
print("图三的经纬度坐标为，计算",utm_to_latlon(p3[0],p3[1]))


















# 求旋转矩阵法2
# # 定义两个点的坐标
# p = np.array([-78.16504048, 54.48358478, 466])
# pose2_in_camera = np.array([-10.29373428, -53.42916504, 466])

# # 计算单位向量
# p_unit = p / np.linalg.norm(p)
# pose2_in_camera_unit = pose2_in_camera / np.linalg.norm(pose2_in_camera)

# # 计算叉积和点积
# v = np.cross(p_unit, pose2_in_camera_unit)
# c = np.dot(p_unit, pose2_in_camera_unit)
# s = np.linalg.norm(v)

# # 构建旋转矩阵
# I = np.eye(3)
# vx = np.array([
#     [0, -v[2], v[1]],
#     [v[2], 0, -v[0]],
#     [-v[1], v[0], 0]
# ])
# R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
# R=np.linalg.inv(R)
# # 打印旋转矩阵
# print(f"旋转矩阵为:\n{R}")

# # 验证
# print(f"验证结果:\n{R @ pose2_in_camera}")