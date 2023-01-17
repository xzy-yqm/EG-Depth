#!/usr/bin/env bash
set -x
DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=2,3 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 35 --lr 0.001 --lrepochs "12,16,18,20,25,30,35:2" --batch_size 8 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/sceneflow/cost_volum_guided  \
    --test_batch_size 8 --resume