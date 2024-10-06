import csv
import folium
from folium.plugins import MousePosition
import tools

# 文件列表
file_list = [
    # '/home/arc/works/review_prj/UAV_slam/datas/uav_track1+90.csv',
    '/home/arc/works/review_prj/UAV_slam/datas/uav_track198+90.csv',
    '/home/arc/works/review_prj/UAV_slam/datas/uav_track390+90.csv',
    # '/home/arc/works/review_prj/UAV_slam/datas/uav_track580+90.csv',
]
true_file='/home/arc/works/review_prj/UAV_slam/src/03.csv'

# 初始化地图对象，初始位置为第一个文件的第一个点
initial_lat, initial_lon = None, None

# 创建地图对象
m = folium.Map(location=[0, 0], zoom_start=15)

# 添加高德地图卫星图层
folium.TileLayer(
    tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    attr='&copy; <a href="http://ditu.amap.com/">高德地图</a>',
    name='高德卫星图'
).add_to(m)


# 存储所有文件的数据
all_data = []

show_num=[]
# 遍历文件列表
for idx, file_path in enumerate(file_list):
    latitudes = []
    longitudes = []
    errors = []
    frame_num=[]
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if 'uav_lon' in row and 'uav_lat' in row:
                lat, lon = tools.wgs84_to_gcj02(float(row['uav_lon']), float(row['uav_lat']))
                errors.append(float(row['error']))
                frame_num.append(int(row['frame_num']))
            latitudes.append(lat)
            longitudes.append(lon)
    show_num.append(frame_num)

    all_data.append((latitudes, longitudes, errors))
# 遍历文件列表


# 读取并处理参考真值文件
ref_latitudes = []
ref_longitudes = []
ref_latlon=[list(),list(),list(),list()]
with open(true_file, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)

    for row in csvreader:
        for i in range(len(show_num)):
            if int(row['num']) in show_num[i]:
                lat, lon = tools.wgs84_to_gcj02(float(row['lon']), float(row['lat']))
                ref_latlon[i].append([lat,lon])


# 添加数据到地图
for idx, (latitudes, longitudes, errors) in enumerate(all_data):
    if initial_lat is None and initial_lon is None:
        initial_lat, initial_lon = latitudes[0], longitudes[0]
        m.location = [initial_lat, initial_lon]

    # 创建误差轨迹图层
    track_layer_name = f'误差轨迹{idx + 1}'
    fg_track = folium.FeatureGroup(name=track_layer_name)
    coordinates = list(zip(latitudes, longitudes))
    folium.PolyLine(coordinates, color='red', weight=2.5, opacity=1).add_to(fg_track)
    fg_track.add_to(m)

    # 创建误差点图层
    points_layer_name = f'误差轨迹{idx + 1}点'
    fg_points = folium.FeatureGroup(name=points_layer_name)
    for lat, lon, error in zip(latitudes, longitudes, errors):
        popup_text = f"Lat: {lat}<br>Lon: {lon}<br>Error: {error}m" if errors else f"Lat: {lat}<br>Lon: {lon}"
        folium.Marker(location=[lat, lon], icon=folium.Icon(color='red'), popup=popup_text).add_to(fg_points)
    fg_points.add_to(m)

    # 创建真值轨迹图层
    true_track_layer_name = f'真值轨迹{idx + 1}'
    fg_true_track = folium.FeatureGroup(name=true_track_layer_name)
    true_coordinates = ref_latlon[idx]
    folium.PolyLine(true_coordinates, color='blue', weight=2.5, opacity=1).add_to(fg_true_track)
    fg_true_track.add_to(m)

    # 创建真值点图层
    true_points_layer_name = f'真值轨迹{idx + 1}点'
    fg_true_points = folium.FeatureGroup(name=true_points_layer_name)
    for true_lat, true_lon in true_coordinates:
        folium.Marker(location=[true_lat, true_lon], icon=folium.Icon(color='blue'), popup=f"Lat: {true_lat}<br>Lon: {true_lon}").add_to(fg_true_points)
    fg_true_points.add_to(m)

# 添加鼠标位置插件
MousePosition().add_to(m)

# 添加点击显示经纬度功能
m.add_child(folium.LatLngPopup())

# 添加图层控制
folium.LayerControl().add_to(m)

# 保存地图为HTML文件
m.save('map.html')