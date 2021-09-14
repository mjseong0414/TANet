import json
import os
import sys
import numpy as np
"""
find 3d boxes x, y, z min max
"""
# json_path = "/home/minjae/JRDB/train_dataset_with_activity/labels/labels_3d"
# file_list = os.listdir(json_path)
# json_files = file_list[2:]
# x_centers = []
# y_centers = []
# z_centers = []
# for i in json_files:
#     with open(json_path + "/" + i ,"r") as f:
#         json_data = json.load(f)
#         for k in range(len(json_data["labels"])):
#             pcd_name = "{0:06d}".format(k) + ".pcd"
#             integer = 0
#             for s in range(len(json_data["labels"][pcd_name])):
#                 x_center = json_data["labels"][pcd_name][s]["box"]["cx"]
#                 y_center = json_data["labels"][pcd_name][s]["box"]["cy"]
#                 z_center = json_data["labels"][pcd_name][s]["box"]["cz"]
#                 x_centers.append(x_center)
#                 y_centers.append(y_center)
#                 if z_center < -15.0:
#                     continue
#                 else:
#                     z_centers.append(z_center)

# max_x_center, min_x_center = max(x_centers), min(x_centers)
# max_y_center, min_y_center = max(y_centers), min(y_centers)
# max_z_center, min_z_center = max(z_centers), min(z_centers)
# print("The range of x_center => {0} ~ {1}".format(min_x_center, max_x_center))
# print("The range of y_center => {0} ~ {1}".format(min_y_center, max_y_center))
# print("The range of z_center => {0} ~ {1}".format(min_z_center, max_z_center))

"""
find points in GT 3D bbox
"""
def camera_to_lidar_roughV1(points, height = None, dimension = None, name=None):
    x_lidar = float(points[2])
    y_lidar = -float(points[0])
    z_lidar = -float(points[1]) + 0.5* float(height)
    xyz_lidar_rough = [str(x_lidar), str(y_lidar), str(z_lidar)]
    return xyz_lidar_rough

# label.txt load
label_dir = "/home/minjae/TANet/pointpillars_with_TANet/second/demo_open3d/label_JRDB/"
label_list = os.listdir(label_dir)
label_list_sort = sorted(label_list)
with open(label_dir + label_list_sort[0], 'r') as f:
    labels = f.readlines()

label = labels[0].split(' ')
dim = label[8:11] # hwl
loc = label[11:14] # x(l), y(h), z(w)
loc_lidar = camera_to_lidar_roughV1(loc, dim[0])

x_range_max = np.array(float(loc[0]) + float(dim[2]))
x_range_min = np.array(float(loc[0]) - float(dim[2]))

y_range_max = np.array(float(loc[1]) + float(dim[0]))
y_range_min = np.array(float(loc[1]) - float(dim[0]))

z_range_max = np.array(float(loc[2]) + float(dim[1]))
z_range_min = np.array(float(loc[2]) - float(dim[1]))

print(str(x_range_min), str(x_range_max))
print(y_range_min, y_range_max)
print(z_range_min, z_range_max)
# print("(x_range_min, x_range_max) = ({0}, {1})", format(str(x_range_min), str(x_range_max)))
# print("(y_range_min, y_range_max) = ({0}, {1})", format(y_range_min, y_range_max))
# print("(x_range_min, z_range_max) = ({0}, {1})", format(z_range_min, z_range_max))

# point cloud load part
points_dir = '/home/minjae/TANet/pointpillars_with_TANet/second/demo_open3d/points/'
points_list = os.listdir(points_dir)
points_lists_sort = sorted(points_list)
points = np.load(points_dir + points_lists_sort[0]).reshape(-1, 4)
valid_points = []
for i in range(points.shape[0]):
    if (x_range_min <= points[i][0] <= x_range_max) and (y_range_min <= points[i][1] <= y_range_max) and (z_range_min <= points[i][2] <= z_range_max):
        valid_points.append(points[i])

