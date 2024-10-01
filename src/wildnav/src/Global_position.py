import csv
import cv2
import haversine as hs
from haversine import Unit
import wildnav.src.superglue_utils
import shutil
import os


class GeoPhoto:
    def __init__(self, filename, photo, geo_top_left, geo_bottom_right):
        self.filename = filename
        self.photo = photo
        self.top_left_coord = geo_top_left
        self.bottom_right_coord = geo_bottom_right

    def __lt__(self, other):
        return self.filename < other.filename

    def __str__(self):
        return "%s; \n\ttop_left_latitude: %f \n\ttop_left_lon: %f \n\tbottom_right_lat: %f \n\tbottom_right_lon %f " % (self.filename, self.top_left_coord[0], self.top_left_coord[1], self.bottom_right_coord[0], self.bottom_right_coord[1])

class GeoLocator:
    def __init__(self, map_path, map_filename):
        self.map_path = map_path
        self.map_filename = map_filename
        self.geo_images_list = None
        self.sub_map_path = os.path.join(map_path, '../sub_map/')
        self.sub_map_csv = os.path.join(self.map_path, '../sub_map/map.csv')

    def csv_read_sat_map(self, filename):
        geo_list = []
        photo_path = self.map_path
        print("opening: ", filename)
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    img = cv2.imread(photo_path + row[0], 0)
                    geo_photo = GeoPhoto(photo_path + row[0], img, (float(row[1]), float(row[2])), (float(row[3]), float(row[4])))
                    geo_list.append(geo_photo)
                    line_count += 1

            print(f'Processed {line_count} lines.')
            geo_list.sort()
            return geo_list
    def filter_and_copy_maps(self,map_csv, start_pose, delta, map_path):
        filtered_maps = []
        # 首先要删除sub_map_path文件夹下的所有文件
        shutil.rmtree(self.sub_map_path, ignore_errors=True)
        # 创建子地图文件夹
        os.makedirs(self.sub_map_path, exist_ok=True)
        with open(map_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                top_left_lat = float(row['Top_left_lat'])
                top_left_lon = float(row['Top_left_lon'])
                bottom_right_lat = float(row['Bottom_right_lat'])
                bottom_right_lon = float(row['Bottom_right_long'])

                if (start_pose[0] - delta <= top_left_lat <= start_pose[0] + delta and
                    start_pose[1] - delta <= top_left_lon <= start_pose[1] + delta) or \
                (start_pose[0] - delta <= bottom_right_lat <= start_pose[0] + delta and
                    start_pose[1] - delta <= bottom_right_lon <= start_pose[1] + delta):
                    filtered_maps.append(row)
                    src_file = os.path.join(map_path, row['Filename'])
                    dst_file = os.path.join(self.sub_map_path, row['Filename'])
                    shutil.copy(src_file, dst_file)


        with open(self.sub_map_csv, 'w', newline='') as csvfile:
            fieldnames = ['Filename', 'Top_left_lat', 'Top_left_lon', 'Bottom_right_lat', 'Bottom_right_long']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in filtered_maps:
                writer.writerow(row)

    def calculate_geo_pose(self,geo_photo, center, features_mean,  shape):
        """
        Calculates the geographical location of the drone image.
        Input: satellite geotagged image, relative pixel center of the drone image, 
        (center with x = 0.5 and y = 0.5 means the located features are in the middle of the sat image)
        pixel coordinatess (horizontal and vertical) of where the features are localted in the sat image, shape of the sat image
        """
        #use ratio here instead of pixels because image is reshaped in superglue    
        latitude = geo_photo.top_left_coord[0] + abs( center[1])  * ( geo_photo.bottom_right_coord[0] - geo_photo.top_left_coord[0])
        longitude = geo_photo.top_left_coord[1] + abs(center[0])  * ( geo_photo.bottom_right_coord[1] - geo_photo.top_left_coord[1])
        
        return latitude, longitude

    def locate_image(self, query_image,suspected_pose, delta_pose):
        self.filter_and_copy_maps(self.map_filename, suspected_pose, delta_pose, self.map_path)
        self.geo_images_list = self.csv_read_sat_map(self.sub_map_csv)
        print(f"{len(self.geo_images_list)} satellite images were loaded.")
        max_features = 10
        located = False
        center = None

        rotations = [0]
        # current_location = self.calculate_geo_pose(self.geo_images_list[297], center, 125, query_image.shape)
        # return current_location
        import time
        start_time = time.time()
        for rot in rotations:
            cv2.imwrite(self.sub_map_path + "1_query_image.png", query_image)
            satellite_map_index_new, center_new, located_image_new, features_mean_new, query_image_new, feature_number = wildnav.src.superglue_utils.match_image(self.sub_map_path)

            if feature_number > max_features :
                satellite_map_index = satellite_map_index_new
                center = center_new
                located_image = located_image_new
                features_mean = features_mean_new
                query_image = query_image_new
                max_features = feature_number
                located = True
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time}")
        print(f"Features matched: {max_features}\n----------------------")

        if center is not None and located:
            print(f'center: {center}')
            print(f'satellite_map_index: {satellite_map_index}')
            current_location = self.calculate_geo_pose(self.geo_images_list[satellite_map_index], center, features_mean, query_image.shape)
            return current_location
        else:
            return None

# # 使用示例
# map_path = "/home/arc/works/review_prj/UAV_slam/wildnav/assets/map/"
# map_filename = "/home/arc/works/review_prj/UAV_slam/wildnav/assets/map/map.csv"
# geo_locator = GeoLocator(map_path, map_filename)

# # 传入查询图片变量
# start_pose=[32.30827647,119.8912811]
# query_image = cv2.imread("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0008.JPG")
# location = geo_locator.locate_image(query_image)
# if location:
#     print(f"Calculated location: {location}")
# else:
#     print("Image not matched.")





if __name__ == "__main__":
    # 使用示例
    map_csv = "/home/arc/works/review_prj/UAV_slam/wildnav/assets/map/map.csv"
    map_path = "/home/arc/works/review_prj/UAV_slam/wildnav/assets/map/"
    suspected_pose = [32.30827647, 119.8912811]
    delta_pose = 0.01

    # 初始化 GeoLocator 使用新的子地图
    geo_locator = GeoLocator(map_path, map_csv)

    # 传入查询图片变量
    query_image = cv2.imread("/mnt/d/Dataset/UAV_VisLoc_dataset/03/drone/03_0008.JPG")
    location = geo_locator.locate_image(query_image,suspected_pose, delta_pose)
    if location:
        print(f"Calculated location: {location}")
    else:
        print("Image not matched.")