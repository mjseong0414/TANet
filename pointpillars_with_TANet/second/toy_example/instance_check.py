import sys
import os

label_path = "/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/training/label_2/"
label_lists = os.listdir(label_path)
instance_counts = 0
#import pdb; pdb.set_trace()
for i in label_lists:
    with open(label_path + i,"r") as f:
        instance_counts += len(f.readlines())

print("The mean of instances per scenes ===> ", str(instance_counts/len(label_lists)))