import torch
from torch import nn
import pdb
import numpy as np
import os

img_path = os.path.abspath("/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/training/velodyne")
image_list = os.listdir(img_path)
for i in image_list:
    bin_name = i.split(".")[0]
    pdb.set_trace()
    np.load(i).tofile(bin_name+".bin")