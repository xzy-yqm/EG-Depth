#!/usr/bin/env bash
set -x
DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=2,3 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 25 --lr 0.001 --lrepochs "10:10" --batch_size 8 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/sceneflow/data_set_bug_fixed_mish  \
    --loadckpt "./checkpoints/sceneflow/data_set_bug_fixed/checkpoint_000019.ckpt" \
    --test_batch_size 2 
    #--loadckpt "./checkpoints/sceneflow/uniform_sample_d256/checkpoint_000019.ckpt" \
