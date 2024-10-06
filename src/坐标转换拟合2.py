import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    """返回二维旋转矩阵"""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def residuals(params, P, P_prime, R_phi_array):
    """
    计算残差向量

    参数:
    - params: 待优化的参数数组 [theta1, theta2, t1x, t1y, t_phix, t_phiy, t2x, t2y]
    - P: 原始点的 Nx2 数组
    - P_prime: 目标点的 Nx2 数组
    - R_phi_array: 每个点对应的已知旋转矩阵，形状为 (N, 2, 2)
    """
    theta1, theta2, t1x, t1y, t_phix, t_phiy, t2x, t2y,k2= params
    R1 = rotation_matrix(theta1)
    R2 = rotation_matrix(theta2)
    t1 = np.array([t1x, t1y])
    t_phi = np.array([t_phix, t_phiy])
    t2 = np.array([t2x, t2y])

    # 应用 R1 和 t1
    intermediate1 =P @ R1.T + t1       # (N,2)

    # 应用 R_phi_i 和 t_phi
    intermediate2 = np.einsum('ijk,ik->ij', R_phi_array, intermediate1) + t_phi  # (N,2)

    # 应用 R2 和 t2
    transformed_P =k2* intermediate2 @ R2.T + t2  # (N,2)

    # 计算残差并展平为一维数组
    residual = (P_prime - transformed_P).ravel()
    return residual

def fit_transformation(P, P_prime, R_phi_array, initial_params=None):
    """
    拟合坐标转换参数

    参数:
    - P: 原始点的 Nx2 数组
    - P_prime: 目标点的 Nx2 数组
    - R_phi_array: 每个点对应的已知旋转矩阵，形状为 (N, 2, 2)
    - initial_params: 初始参数数组（可选）

    返回:
    - result: 最小二乘拟合结果
    """
    if initial_params is None:
        # 初始化参数：[theta1, theta2, t1x, t1y, t_phix, t_phiy, t2x, t2y,k2]
        initial_params = np.zeros(9)

    result = least_squares(
        residuals,
        initial_params,
        args=(P, P_prime, R_phi_array),
        method='lm'  # Levenberg-Marquardt 算法
    )
    return result

def generate_synthetic_data(N=100, noise_level=0.01):
    """
    生成合成数据用于测试

    参数:
    - N: 点的数量
    - noise_level: 噪声水平

    返回:
    - P: 原始点的 Nx2 数组
    - P_prime: 目标点的 Nx2 数组
    - R_phi_array: 每个点对应的旋转矩阵，形状为 (N, 2, 2)
    - true_params: 真实的参数数组
    """
    # 随机生成原始点
    P = np.random.uniform(-1, 1, (N, 2))

    # 为每个点生成一个随机的 R_phi
    phi_array = np.deg2rad(np.random.uniform(-45, 45, N))  # 例如，旋转角度在 -45 到 45 度之间
    R_phi_array = np.array([rotation_matrix(phi) for phi in phi_array])  # (N, 2, 2)

    # 定义真实参数
    theta1_true = np.deg2rad(10)   # 10度
    theta2_true = np.deg2rad(-20)  # -20度
    t1_true = np.array([0.5, -0.3])
    t_phi_true = np.array([0.2, 0.4])
    t2_true = np.array([10,30])

    true_params = [theta1_true, theta2_true, *t1_true, *t_phi_true, *t2_true]

    # 计算 P_prime
    R1 = rotation_matrix(theta1_true)    # (2,2)
    R2 = rotation_matrix(theta2_true)    # (2,2)

    # 应用 R1 和 t1
    intermediate1 = P @ R1.T + t1_true          # (N,2)

    # 应用 R_phi_i 和 t_phi
    intermediate2 = np.einsum('ijk,ik->ij', R_phi_array, intermediate1) + t_phi_true  # (N,2)

    # 应用 R2 和 t2
    transformed_P = intermediate2 @ R2.T + t2_true  # (N,2)

    # 添加噪声
    P_prime = transformed_P + noise_level * np.random.randn(N, 2)

    return P, P_prime, R_phi_array, true_params
import csv

def load_data_from_csv(file_path):
    """
    从CSV文件中加载数据

    参数:
    - file_path: CSV文件路径

    返回:
    - P: 原始点的 Nx2 数组 (u, v)
    - P_prime: 目标点的 Nx2 数组 (true_posex, true_posey)
    """
    P = []
    P_prime = []
    phi_array = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            P.append([float(row['u']), float(row['v'])])
            P_prime.append([float(row['true_posex']), float(row['true_posey'])])
            phi_matrix = rotation_matrix(np.deg2rad(float(row['phi'])))
            phi_array.append(phi_matrix)
    
    P=np.array(P)
    P_prime=np.array(P_prime)
    phi_array=np.array(phi_array)
    return P, P_prime,phi_array
def main():
    # 生成合成数据
    # P, P_prime, R_phi_array, true_params = generate_synthetic_data(N=200, noise_level=0.01)
    P, P_prime, R_phi_array=load_data_from_csv('/home/arc/works/review_prj/UAV_slam/src/uav_track.csv')
    # print("真实参数:")
    # print(f"Theta1 (deg): {np.rad2deg(true_params[0]):.2f}")
    # print(f"Theta2 (deg): {np.rad2deg(true_params[1]):.2f}")
    # print(f"t1: {true_params[2:4]}")
    # print(f"t_phi: {true_params[4:6]}")
    # print(f"t2: {true_params[6:8]}")

    # 执行拟合
    result = fit_transformation(P, P_prime, R_phi_array)

    # # 提取拟合参数
    fitted_theta1, fitted_theta2, fitted_t1x, fitted_t1y, fitted_t_phix, fitted_t_phiy, fitted_t2x, fitted_t2y,k2= result.x
    print(fitted_theta1, fitted_theta2, fitted_t1x, fitted_t1y, fitted_t_phix, fitted_t_phiy, fitted_t2x, fitted_t2y,k2)
    # fitted_theta1, fitted_theta2, fitted_t1x, fitted_t1y, fitted_t_phix, fitted_t_phiy, fitted_t2x, fitted_t2y,k2=-1.2175891004503503,-0.16180359775692835,18758.608364114167,5264.337166904729,2525.971934191457,3208.5992371944594,-2234.0566678496252,900.1388449437294,0.11537983437500669
    R1_fitted = rotation_matrix(fitted_theta1)
    R2_fitted = rotation_matrix(fitted_theta2)
    t1_fitted = np.array([fitted_t1x, fitted_t1y])
    t_phi_fitted = np.array([fitted_t_phix, fitted_t_phiy])
    t2_fitted = np.array([fitted_t2x, fitted_t2y])

    print("\n拟合参数:")
    print(f"Theta1 (deg): {np.rad2deg(fitted_theta1):.2f}")
    print(f"Theta2 (deg): {np.rad2deg(fitted_theta2):.2f}")
    print(f"t1: {t1_fitted}")
    print(f"t_phi: {t_phi_fitted}")
    print(f"t2: {t2_fitted}")

    # 计算拟合后的 P_prime
    # 应用 R1 和 t1
    intermediate1_fitted = P @ R1_fitted.T + t1_fitted        # (N,2)

    # 应用 R_phi_i 和 t_phi
    intermediate2_fitted = np.einsum('ijk,ik->ij', R_phi_array, intermediate1_fitted) + t_phi_fitted  # (N,2)

    # 应用 R2 和 t2
    transformed_P_fitted =k2*intermediate2_fitted @ R2_fitted.T + t2_fitted  # (N,2)

    # 绘制结果
    plt.figure(figsize=(8, 8))
    plt.scatter(P[:, 0], P[:, 1], label='Original P', alpha=0.5)
    plt.scatter(P_prime[:, 0], P_prime[:, 1], label='Target P\'', alpha=0.5)
    plt.scatter(transformed_P_fitted[:, 0], transformed_P_fitted[:, 1], label='Fitted P\'', alpha=0.5)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('坐标转换拟合结果')
    plt.axis('equal')
    plt.show()

    # 打印拟合误差
    # final_residual = np.linalg.norm(result.fun)
    # print(f"\n最终拟合误差（L2 范数）: {final_residual:.4f}")


def get_pose(uv_pose,phi):
    fitted_theta1, fitted_theta2, fitted_t1x, fitted_t1y, fitted_t_phix, fitted_t_phiy, fitted_t2x, fitted_t2y,k2=-1.2175891004503503,-0.16180359775692835,18758.608364114167,5264.337166904729,2525.971934191457,3208.5992371944594,-2234.0566678496252,900.1388449437294,0.11537983437500669
        
    R1_fitted = rotation_matrix(fitted_theta1)
    R2_fitted = rotation_matrix(fitted_theta2)
    t1_fitted = np.array([fitted_t1x, fitted_t1y])
    t_phi_fitted = np.array([fitted_t_phix, fitted_t_phiy])
    t2_fitted = np.array([fitted_t2x, fitted_t2y])

    phi_matrix = rotation_matrix(np.deg2rad(phi))
    
    # 应用 R1 和 t1
    intermediate1_fitted = R1_fitted@uv_pose  + t1_fitted  # (N,2)
    
    # 应用 R_phi 和 t_phi
    intermediate2_fitted = phi_matrix@intermediate1_fitted + t_phi_fitted  # (2,)
    
    # 应用 R2 和 t2
    transformed_P_fitted = k2 *R2_fitted@intermediate2_fitted  + t2_fitted  # (N,2)
    
    return transformed_P_fitted

if __name__ == "__main__":
    main()
    P=get_pose([-1523.6279182434082,-8948.960876464844],-39.895051)
    P=get_pose([333.8728332519531,-1663.7562866210938],121.09463)
    print(P)
    # -847.1599174691364,612.4102629264817
