import pickle
import numpy as np

pickle_name = "/home/spalab/jrdb_3dteam/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_train.pkl"


# nps = np.array([473., 254.,  16., 363., 187.,  97., 563.,  71., 311., 215.,  16.,
#        286., 347., 783., 139.,  26.])
# for i in nps:
#     print(i)
    

with open(pickle_name, "rb") as f:
    data_val = pickle.load(f)

import pdb; pdb.set_trace()
# invalid_gt_val = 0
# for i in range(len(data_val)):
#     for pt in data_val[i]["annos"]["num_points_in_gt"]:
#         if pt < 10:
#             invalid_gt_val += 1

# print(invalid_gt_val)
# invalid_gt_val = 0
# for i in range(len(data_val)):
#     for k in range(len(data_val[i]["annos"]["location"])):
#         if -25.0 > float(data_val[i]["annos"]["location"][k][0]) or float(data_val[i]["annos"]["location"][k][0]) > 25.0:
#             invalid_gt_val += 1
#         elif -25.0 > float(data_val[i]["annos"]["location"][k][1]) or float(data_val[i]["annos"]["location"][k][1]) > 25.0:
#             invalid_gt_val += 1
#         else:
#             break

# print(invalid_gt_val)

'''
with open("/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_to_kitti_infos_train.pkl", "rb") as f:
    data_train = pickle.load(f)

invalid_gt_train = 0
for i in range(len(data_train)):
    for a in data_train[i]["annos"]["num_points_in_gt"]:
        if a < 10:
            invalid_gt_train += 1

print(invalid_gt_train)

invalid_gt_train = 0
for i in range(len(data_train)):
    for k in range(len(data_train[i]["annos"]["location"])):
        if -25.0 > float(data_train[i]["annos"]["location"][k][0]) or float(data_train[i]["annos"]["location"][k][0]) > 25.0:
            invalid_gt_train += 1
        elif -25.0 > float(data_train[i]["annos"]["location"][k][1]) or float(data_train[i]["annos"]["location"][k][1]) > 25.0:
            invalid_gt_train += 1
        else:
            break
print(invalid_gt_train)
'''