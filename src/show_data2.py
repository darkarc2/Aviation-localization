from flask import Flask, render_template, request, jsonify
import folium
import csv
import tools

app = Flask(__name__)

# 初始化地图对象
m = folium.Map(location=[32.304626729999995, 119.8968847], zoom_start=15)
folium.TileLayer(
    tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
    attr='&copy; <a href="http://ditu.amap.com/">高德地图</a>',
    name='高德卫星图'
).add_to(m)
folium.LayerControl().add_to(m)

# 保存初始地图
m.save('templates/map.html')

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    lat = data['lat']
    lon = data['lon']
    folium.Marker([lat, lon]).add_to(m)
    m.save('templates/map.html')
    return jsonify(status="success")

if __name__ == '__main__':
    app.run(debug=True)