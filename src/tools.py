import pyproj
import numpy as np
import csv

#经纬度坐标转换为UTM坐标
def latlon_to_utm(lon, lat, zone=50):
    wgs84 = pyproj.CRS('EPSG:4326')  # 定义WGS84坐标系
    utm = pyproj.CRS(f'EPSG:326{zone}')  # 定义UTM坐标系
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

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

def read_csv(file_path):
    frames = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            frames.append({
                'num': int(row['num']),
                'filename': row['filename'],
                'date': row['date'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'height': float(row['height']),
                'Omega': float(row['Omega']),
                'Kappa': float(row['Kappa']),
                'Phi1': float(row['Phi1']),
                'Phi2': float(row['Phi2'])
            })
    return frames

def calculate_errors(frames, uav):
    errors = []
    start_num = 0
    frames = frames[start_num:]
    start_pose_utm = list(latlon_to_utm(frames[0]['lon'],frames[0]['lat']))
    end_pose_utm = list(latlon_to_utm(frames[len( uav.frames)]['lon'],frames[len( uav.frames)]['lat']))
    for i in range(len( uav.frames)):
        frame = frames[i]
        uav_pose = uav.frames[i].get_pose()
        uav_pose = uav_pose[0:2]+np.array(start_pose_utm)
        uav_lat, uav_lon = utm_to_latlon(uav_pose[0], uav_pose[1])
        
        lat_error = frame['lat'] - uav_lat
        lon_error = frame['lon'] - uav_lon
        
        frame_utm_x, frame_utm_y = latlon_to_utm(frame['lon'],frame['lat'])
        uav_utm_x, uav_utm_y = latlon_to_utm(uav_lon,uav_lat)
        
        distance_error = ((frame_utm_x - uav_utm_x) ** 2 + (frame_utm_y - uav_utm_y) ** 2) ** 0.5
        
        errors.append({
            'frame_num': frame['num'],
            'lat_error': lat_error,
            'lon_error': lon_error,
            'distance_error': distance_error
        })
    errors.append({
        'frame_num': 'start---to---end',
        'lat_error': end_pose_utm[0] - start_pose_utm[0],
        'lon_error': end_pose_utm[1] - start_pose_utm[1],
        'distance_error': ((end_pose_utm[0] - start_pose_utm[0]) ** 2 + (end_pose_utm[1] - start_pose_utm[1]) ** 2) ** 0.5
    })
    return errors














def test():
	# 经纬度坐标
	lat1, lon1 = 32.30462673, 119.8968847
	lat2, lon2 = 32.30513666, 119.8960711
	lat3, lon3=32.30566378,119.8952804
	zone=50

	# 将经纬度坐标转换为UTM坐标
	x1, y1 = latlon_to_utm(lon1, lat1)
	x2, y2 = latlon_to_utm(lon2, lat2)
	x3, y3 = latlon_to_utm(lon3, lat3)

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

		# 将UTM坐标转换回经纬度坐标
	lat1_back, lon1_back = utm_to_latlon(x1, y1)
	lat2_back, lon2_back = utm_to_latlon(x2, y2)
	lat3_back, lon3_back = utm_to_latlon(x3, y3)

	print(f"点1的经纬度坐标: ({lat1_back}, {lon1_back})")
	print(f"点2的经纬度坐标: ({lat2_back}, {lon2_back})")
	print(f"点3的经纬度坐标: ({lat3_back}, {lon3_back})")


# test()