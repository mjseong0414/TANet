import os
import glob

# 바꿀 이미지가 위치한 폴더
img_path = os.path.abspath("/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI_test/training/velodyne")
image_list = os.listdir(img_path)

for i in image_list:
    bin_name = i.split(".")[0] + ".bin"
    # import pdb; pdb.set_trace()
    os.rename(img_path + "/" + i, img_path + "/" + bin_name)