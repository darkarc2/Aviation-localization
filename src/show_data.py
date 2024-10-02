import csv
import folium
from folium.plugins import MousePosition
import tools

# 读取第一份CSV文件
latitudes1 = []
longitudes1 = []

with open('/home/arc/works/review_prj/UAV_slam/src/03.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    count = 0
    for row in csvreader:
        lat, lon = tools.wgs84_to_gcj02(float(row['lon']), float(row['lat']))
        latitudes1.append(lat)
        longitudes1.append(lon)
        count += 1
        if count >= 90:
            break

# 读取第二份CSV文件
latitudes2 = []
longitudes2 = []
errors = []

with open('/home/arc/works/review_prj/UAV_slam/src/uav_track.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        lat, lon = tools.wgs84_to_gcj02(float(row['uav_lon']), float(row['uav_lat']))
        latitudes2.append(lat)
        longitudes2.append(lon)
        errors.append(float(row['error']))

# 创建地图对象，初始位置为第一个点
m = folium.Map(location=[latitudes1[0], longitudes1[0]], zoom_start=15)

# 添加高德地图卫星图层
folium.TileLayer(
    tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    attr='&copy; <a href="http://ditu.amap.com/">高德地图</a>',
    name='高德卫星图'
).add_to(m)

# 创建第一份CSV文件的轨迹图层
fg1_track = folium.FeatureGroup(name='轨迹1')
coordinates1 = list(zip(latitudes1, longitudes1))
folium.PolyLine(coordinates1, color='blue', weight=2.5, opacity=1).add_to(fg1_track)
fg1_track.add_to(m)

# 创建第一份CSV文件的点图层
fg1_points = folium.FeatureGroup(name='轨迹1点')
for lat, lon in coordinates1:
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='blue')).add_to(fg1_points)
fg1_points.add_to(m)

# 创建第二份CSV文件的轨迹图层
fg2_track = folium.FeatureGroup(name='轨迹2')
coordinates2 = list(zip(latitudes2, longitudes2))
folium.PolyLine(coordinates2, color='red', weight=2.5, opacity=1).add_to(fg2_track)
fg2_track.add_to(m)

# 创建第二份CSV文件的点图层
fg2_points = folium.FeatureGroup(name='轨迹2点')
for lat, lon, error in zip(latitudes2, longitudes2, errors):
    popup_text = f"Lat: {lat}<br>Lon: {lon}<br>Error: {error}m"
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='red'), popup=popup_text).add_to(fg2_points)
fg2_points.add_to(m)

# 添加鼠标位置插件
MousePosition().add_to(m)

# 添加点击显示经纬度功能
m.add_child(folium.LatLngPopup())

# 添加图层控制
folium.LayerControl().add_to(m)

# 保存地图为HTML文件
m.save('map.html')