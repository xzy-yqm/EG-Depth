#!/usr/bin/env bash
set -x
#DATAPATH="/chengyh/xzy_ws/data/"
DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=2,3 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 6 --maxdisp 576 \
    --model cfnet --logdir ./checkpoints/sceneflow/maxdisp_576  \
    --test_batch_size 4 
    #--resume 
