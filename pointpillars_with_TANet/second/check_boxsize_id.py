import sys
import os
import json
import numpy as np
import pdb

target_scene = os.listdir("data/JRDB_to_KITTI/JRDB_JSON/labels/labels_3d/")
target_id = 20
total_id_num = 243
init = 1
target_h_ = 0
target_l_ = 0
target_w_ = 0

label_db = []
scene_id_tracking = []
scene_id_size_tracking = []

id_box_db = np.zeros((total_id_num,4), dtype=float)
id_idx = 0

for i in range(len(target_scene)):
    
    if target_scene[i] == '.bytes-cafe-2019-02-07_0.json.swp':
        continue

    with open("data/JRDB_to_KITTI/JRDB_JSON/labels/labels_3d/" + target_scene[i], "r") as f:
        label_3d_json = json.load(f)
    

    label_3d_dict = label_3d_json['labels']
    for key in label_3d_dict:
        #print(type(label_3d_dict[key]))
        for idx in range(len(label_3d_dict[key])):
            frame_3d_dict = label_3d_dict[key][idx]
            label_id = int(frame_3d_dict['label_id'].split(":")[-1])
            
            if label_id not in label_db:
                label_db.append(label_id)
                
                id_box_db[id_idx,0] = label_id
                id_box_db[id_idx,1] = frame_3d_dict['box']['h']
                id_box_db[id_idx,2] = frame_3d_dict['box']['l']
                id_box_db[id_idx,3] = frame_3d_dict['box']['w']
                id_idx+=1
            
            if label_id == target_id:
                scene_name = target_scene[i]
                scene_idx = frame_3d_dict['file_id'].split(".")[0] + '.txt'
                scene_id_tracking.append(scene_name + ' ' + scene_idx + '\n')
                target_h, target_l, target_w = frame_3d_dict['box']['h'], frame_3d_dict['box']['l'], frame_3d_dict['box']['w']
                #print(f"h : {target_h}, l : {target_l}, w : {target_w}")
                #pdb.set_trace()

                """
                if init:
                    init = 0
                    target_h_ = target_h
                    target_l_ = target_l
                    target_w_ = target_w
                    print(f"height: {target_h_}, length: {target_l_}, width: {target_w_}")
                """

                if (target_h_ != target_h) | (target_l_ != target_l) | (target_w_ != target_w):
                    target_h_ = target_h
                    target_l_ = target_l
                    target_w_ = target_w
                    scene_id_size_tracking.append(scene_name + ' ' + str(target_h_) + ' ' + str(target_l_) + ' ' + str(target_w_) + '\n')
                    #print("No same value")
            
        
    #sys.exit()
#print(id_idx)
print(id_box_db.max(axis=0))
print(id_box_db.min(axis=0))
print(f"h : {target_h_}, l : {target_l_}, w : {target_w_}")
#pdb.set_trace()
#print(len(scene_id_tracking))

with open("pesron_" + str(target_id) + "_tracking.txt", "w") as file:
    file.writelines(scene_id_tracking)

with open("pesron_" + str(target_id) + "_size_tracking.txt", "w") as file:
    file.writelines(scene_id_size_tracking)