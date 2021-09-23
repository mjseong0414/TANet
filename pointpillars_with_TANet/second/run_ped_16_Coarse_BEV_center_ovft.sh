#! /bin/bash
###python create_data.py create_kitti_info_file --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_reduced_point_cloud --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_groundtruth_database --data_path=/data2/zheliu/Kitti/object

# CUDA_VISIBLE_DEVICES=3 python ./pytorch/train_JRDB.py train --config_path=./configs/tanet/ped_cycle/minjae/xyres_16_JRDB_BEV_center_ovft.proto --model_dir=./Model_Path/BEV_center_ovft  --refine_weight 2 --bev_target center --l_fixed 0.0 --w_fixed 0.0

CUDA_VISIBLE_DEVICES=3 python ./pytorch/train_JRDB.py evaluateV2 --config_path=./configs/tanet/ped_cycle/minjae/xyres_16_JRDB_BEV_center_ovft.proto --model_dir=./Model_Path/BEV_center_ovft --log_txt log_evaluate.txt --bev_target center --l_fixed 0.0 --w_fixed 0.0
