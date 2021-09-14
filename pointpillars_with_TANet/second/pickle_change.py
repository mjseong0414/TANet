import pickle
import sys
sys.path.append("/home/minjae/TANet/pointpillars_with_TANet/second/data/")

f = open("/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_to_kitti_infos_train_origin.pkl", 'rb')
data = pickle.load(f)
import pdb; pdb.set_trace()

f = open("/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_to_kitti_infos_val_origin.pkl", 'rb')
data = pickle.load(f)
import pdb; pdb.set_trace()


for i in range(len(data)):
    before_str = data[i]['velodyne_path']
    after_str  = before_str.replace('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/', '')
    data[i]['velodyne_path'] = '/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/' + after_str
    #import pdb; pdb.set_trace()

    before_str = data[i]['img_path']
    after_str  = before_str.replace('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/', '')
    data[i]['img_path'] = '/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/' + after_str

with open('/home/minjae/TANet/pointpillars_with_TANet/second/data/JRDB_to_KITTI/JRDB_to_kitti_infos_val.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


'''good
1. 일단 JRDB_to_kitti_infos_val이게 없어야 맞는거
2. FileNotFoundError '/home/minjae/TANet/pointpil 이게 문제인거네
학습때 잘못됐던게 영향이
한영키좀 부탁드립니다용
잠시만

with open('JRDB_to_kitti_infos_.pkl', 'wb')이거 이게 맞나?
아니 그거말고
지금 경로가 
틀렸음
왼쪽 이거봐바 dump가 다른곳에 되고있자나
이렇게 해야지
오우 아까아닌가
ㄳㄳㅈㅅ ㅋㅋㅋ
굳
'''