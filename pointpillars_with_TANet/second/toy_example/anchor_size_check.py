import json
import os
import sys
import numpy as np
"""
find 3d boxes w, l, h mean value
"""
json_path = "/home/minjae/JRDB/train_dataset_with_activity/labels/labels_3d"
file_list = os.listdir(json_path)
json_files = file_list[2:]
heights = []
lengths = []
widths = []
for i in json_files:
    with open(json_path + "/" + i ,"r") as f:
        json_data = json.load(f)
        for k in range(len(json_data["labels"])):
            pcd_name = "{0:06d}".format(k) + ".pcd"
            integer = 0
            for s in range(len(json_data["labels"][pcd_name])):
                height = json_data["labels"][pcd_name][s]["box"]["h"]
                length = json_data["labels"][pcd_name][s]["box"]["l"]
                width = json_data["labels"][pcd_name][s]["box"]["w"]
                heights.append(height)
                lengths.append(length)
                widths.append(width)

height_mean = sum(heights) / len(heights)
lengths_mean = sum(lengths) / len(lengths)
widths_mean = sum(widths) / len(widths)

print("The mean of 3D GTbox's height => {0}".format(height_mean))
print("The mean of 3D GTbox's length => {0}".format(lengths_mean))
print("The mean of 3D GTbox's width => {0}".format(widths_mean))