def extract_pose_from_homography(self,H):
	# Camera intrinsic matrix (assuming fx = fy = 1, cx = cy = 0 for simplicity)
	# K = np.eye(3)
	#  # 分解单应矩阵 H
	a=3976/35.9
	b=2652/24
	f=35

	K = np.array([
		[f*a, 0, 3976//2],
		[0, f*b, 2652//2],
		[0, 0, 1]
	])

	# Decompose the homography matrix
	_, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

	# Select the first solution (you may need to validate which solution is correct)
	R = Rs[0]
	T = Ts[0]
	T*=412.24953990779335
	# T*=3742.9108374078

	# Calculate rotation angle (in degrees)
	theta = np.arctan2(R[1, 0], R[0, 0]) * (180.0 / np.pi)

	# Calculate translation (in pixels)
	tx, ty = T[0], T[1]

	return theta, tx, ty