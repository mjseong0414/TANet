#! /bin/bash
###python create_data.py create_kitti_info_file --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_reduced_point_cloud --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_groundtruth_database --data_path=/data2/zheliu/Kitti/object

# CUDA_VISIBLE_DEVICES=0 python ./pytorch/train_JRDB.py train --config_path=./configs/tanet/ped_cycle/minjae/xyres_16_JRDB_Aug_Ori.proto --model_dir=./Model_Path/Coarse_Only  --refine_weight 2

CUDA_VISIBLE_DEVICES=0 python ./pytorch/train_JRDB.py evaluateV2 --config_path=./configs/tanet/ped_cycle/minjae/xyres_16_JRDB_Aug_Ori.proto --model_dir=./Model_Path/Coarse_Only --log_txt 'log_evaluate.txt'
