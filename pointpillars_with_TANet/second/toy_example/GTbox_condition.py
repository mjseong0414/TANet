import json
import os
import sys
import numpy as np
"""
GTbox count in specific range : x[-25, 25], y[-25, 25] and over the 10 points
"""
json_path = "/home/minjae/JRDB/train_dataset_with_activity/labels/labels_3d"
file_list = os.listdir(json_path)
json_files = file_list[2:]
x_centers = []
y_centers = []
z_centers = []
all_GTbox_counts = 0
GTbox_counts = 0
for i in json_files:
    with open(json_path + "/" + i ,"r") as f:
        json_data = json.load(f)
        for k in range(len(json_data["labels"])):
            pcd_name = "{0:06d}".format(k) + ".pcd"
            integer = 0
            for s in range(len(json_data["labels"][pcd_name])):
                x_center = json_data["labels"][pcd_name][s]["box"]["cx"]
                y_center = json_data["labels"][pcd_name][s]["box"]["cy"]
                z_center = json_data["labels"][pcd_name][s]["box"]["cz"]
                num_points_in_gt = json_data["labels"][pcd_name][s]["attributes"]["num_points"]

                all_GTbox_counts += 1
                if -25.0 < x_center and x_center < 25.0 and -25.0 < y_center and y_center < 25 and 10<= num_points_in_gt:
                    x_centers.append(x_center)
                    y_centers.append(y_center)
                    z_centers.append(z_center)
                    GTbox_counts += 1
                    
print("All GTbox counts => ", all_GTbox_counts)
print("The number of GTboxs in -25 < x < 25, -25 < y < 25 => ", GTbox_counts)
max_x_center, min_x_center = max(x_centers), min(x_centers)
max_y_center, min_y_center = max(y_centers), min(y_centers)
max_z_center, min_z_center = max(z_centers), min(z_centers)
print("The range of x_center => {0} ~ {1}".format(min_x_center, max_x_center))
print("The range of y_center => {0} ~ {1}".format(min_y_center, max_y_center))
print("The range of z_center => {0} ~ {1}".format(min_z_center, max_z_center))