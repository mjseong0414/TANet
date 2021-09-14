import json
import os
import sys
import numpy as np
"""
GTbox count in specific range : x[-25, 25], y[-25, 25] and over the 10 points
So We have to filtering when we make demo
"""
# gt_bboxes
label_dir = '/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/training/label_2/'
label_list = os.listdir(label_dir)
label_lists_sort = sorted(label_list)

idx = 0
for i in range(len(label_lists_sort)):
    new_label = []
    contents = None
    with open(label_dir + label_lists_sort[i], "r") as f:
        contents = f.readlines()
        for k in range(len(contents)):
            contents_split = contents[k].split(' ') # contents_split[11:14] = x, y, z
            if -3.0 <= float(contents_split[11]) <= 3.0 and -3.0 <= float(contents_split[12]) <= 3.0 and int(contents_split[15]) >= 10:
                with open('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/training/label_2_filtered_3m/%06d.txt'%(idx), 'a') as p:
                    p.write(contents[k])
    idx += 1
print("End")