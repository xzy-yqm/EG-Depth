#!/usr/bin/env bash
set -x
DATAPATH="/data/kitti_completion/"
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset kitti_completion \
    --datapath $DATAPATH --trainlist ./filenames/kitti_completion_train.txt --testlist ./filenames/kitti_completion_val.txt \
    --epochs 30 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 8 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/kitti_completion/sgm_sparse_feature_guided_d256_disparity  \
    --resume --test_batch_size 2
