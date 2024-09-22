import numpy as np

# 原始坐标和目标坐标
original_coord = np.array([-1245, -976])
target_coord = np.array([-159, -895])

# 计算旋转角度
angle = np.arctan2(target_coord[1], target_coord[0]) - np.arctan2(original_coord[1], original_coord[0])

# 构建旋转矩阵
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
print(R)

# 旋转后的坐标
rotated_coord = R.dot(original_coord)

# 乘以系数k
k = np.linalg.norm(target_coord) / np.linalg.norm(rotated_coord)
scaled_coord = k * rotated_coord
print(k)
# 验证程序
def verify_coordinates(scaled_coord, target_coord, tolerance=1e-5):
    return np.allclose(scaled_coord, target_coord, atol=tolerance)

# 打印结果
print("旋转后的坐标:", rotated_coord)
print("乘以系数k后的坐标:", scaled_coord)
print("验证结果:", verify_coordinates(scaled_coord, target_coord))

# 验证失败信息
if not verify_coordinates(scaled_coord, target_coord):
    print("验证失败: 旋转后的坐标不接近目标坐标")