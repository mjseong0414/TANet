import pickle
import sys
sys.path.append("/home/minjae/TANet/pointpillars_with_TANet/second/data/")

f = open("/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_to_kitti_infos_val.pkl", 'rb')
data = pickle.load(f)
import pdb; pdb.set_trace()

# f = open("/home/joon/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_to_kitti_infos_val_origin.pkl", 'rb')
# data = pickle.load(f)
# import pdb; pdb.set_trace()


for i in range(len(data)):
    before_str = data[i]['velodyne_path']
    after_str  = before_str.replace('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/', '')
    data[i]['velodyne_path'] = '/home/spalab/jrdb_3dteam/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/' + after_str
    #import pdb; pdb.set_trace()

    before_str = data[i]['img_path']
    after_str  = before_str.replace('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/', '')
    data[i]['img_path'] = '/home/spalab/jrdb_3dteam/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/' + after_str

with open('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_val.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)