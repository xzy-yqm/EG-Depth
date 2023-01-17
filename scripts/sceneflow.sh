#!/usr/bin/env bash
set -x

DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=2,3 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 8 --maxdisp 512 \
    --model cfnet --logdir ./checkpoints/sceneflow/full_net_no_conv_gsm_modify_d512_k2_1_c2_10_rgb0_1.txt  \
    --test_batch_size 4 
