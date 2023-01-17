#!/usr/bin/env bash
set -x
DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=2,3 python robust_sceneflow.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --batch_size 1 --test_batch_size 1 \
    --testlist ./filenames/sceneflow_test.txt --maxdisp 256 \
    --epochs 1 --lr 0.001  --lrepochs "300:10" \
    --loadckpt "/home/zjlab/stereo_match/CFNet/checkpoints/sceneflow/cost_volum_guided/checkpoint_000034.ckpt" \
    --model cfnet --logdir ./checkpoints/sceneflow_origin_abstudy_test


#checkpoints/sceneflow/cost_volum_guided/
#/home/zjlab/stereo_match/CFNet_bakup/checkpoints/sceneflow_pretraining.ckpt