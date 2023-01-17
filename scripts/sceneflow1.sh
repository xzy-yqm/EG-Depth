#!/usr/bin/env bash
set -x
DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=3 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train1.txt --testlist ./filenames/sceneflow_test1.txt \
    --epochs 10000 --lr 0.001 --lrepochs "6000,8000,9000,10000:2" --batch_size 1 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/sceneflow/stereo_lidar_gsm_sparse_feature  --save_freq 1000 \
    --test_batch_size 1
