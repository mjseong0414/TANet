#! /bin/bash
###python create_data.py create_kitti_info_file --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_reduced_point_cloud --data_path=/data2/zheliu/Kitti/object
###python create_data.py create_groundtruth_database --data_path=/data2/zheliu/Kitti/object

CUDA_VISIBLE_DEVICES=0 python ./pytorch/train.py train --config_path=./configs/tanet/ped_cycle/xyres_16.proto --model_dir=./train_16_ped_cycle_tanet_kitti_ovft  --refine_weight 5

CUDA_VISIBLE_DEVICES=0 python ./pytorch/train.py evaluate --config_path=./configs/tanet/ped_cycle/xyres_16.proto --model_dir=./train_16_ped_cycle_tanet_kitti_ovft
