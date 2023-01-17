#!/usr/bin/env bash
set -x
#DATAPATH="/data/kitti_completion/"
DATAPATH="/mnt/nas/kitti_completion"
CUDA_VISIBLE_DEVICES=0 python3 main_test.py --dataset kitti_completion \
    --datapath $DATAPATH --trainlist ./filenames/kitti_completion_train.txt --testlist ./filenames/kitti_completion_val.txt \
    --epochs 1 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 8 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/test  \
    --test_batch_size 1 --loadckpt ./checkpoints/kitti_completion.ckpt 
