import os 
import sys
import numpy as np

label_path = "/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/example"
labels = os.listdir(label_path)
annotations = {}
annotations.update({
    'name': [],
    'truncated': [],
    'occluded': [],
    'alpha': [],
    'bbox': [],
    'dimensions': [],
    'location': [],
    'rotation_y': [],
    'num_points_in_gt': []
})
# line_num = 0
# for i in labels:
#     with open(label_path + "/" + i, "r") as f:
#         lines = f.readlines()
#         line_num +=1
    
    
#     for k in range(len(lines)-1):
#         if -25.0 > float(lines[k].split(" ")[11]) or float(lines[k].split(" ")[11]) > 25.0:
#             print(line_num)
#             print(float(lines[k].split(" ")[11]))
#             import pdb; pdb.set_trace()
del_counts = 0
new_lines = []
for i in labels:
    with open(label_path + "/" + i, "r") as f:
        lines = f.readlines()
    
    for k in range(len(lines)-1):
        if int(lines[k].split(" ")[15]) >= 10:
            if (-25.0 < float(lines[k].split(" ")[11]) < 25.0) and (-25.0 < float(lines[k].split(" ")[12]) < 25.0):
                new_lines.append(lines[k])
                del_counts +=1
    
    content = [line.strip().split(' ') for line in new_lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]] # original = height width length, change = length, height, width
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    annotations['num_points_in_gt'] = np.array(
        [float(x[15]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 17:  # have score
        annotations['score'] = np.array([float(x[16]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    
import pdb; pdb.set_trace()