import pyproj
import numpy as np

#经纬度坐标转换为UTM坐标
def latlon2utm(lon,lat, zone=50):
	wgs84 = pyproj.CRS('EPSG:4326')  # 定义WGS84坐标系
	utm = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')  # 定义ENU坐标系,Zone
	# utm = pyproj.CRS(f'EPSG:326{zone}')
	x, y = pyproj.transform(wgs84, utm, lon, lat)
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

def test():
	# 经纬度坐标
	lat1, lon1 = 32.30462673, 119.8968847
	lat2, lon2 = 32.30513666, 119.8960711
	lat3, lon3=32.30566378,119.8952804
	zone=50

	# 将经纬度坐标转换为UTM坐标
	x1, y1 = latlon2utm(lon1, lat1)
	x2, y2 = latlon2utm(lon2, lat2)
	x3, y3 = latlon2utm(lon3, lat3)

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