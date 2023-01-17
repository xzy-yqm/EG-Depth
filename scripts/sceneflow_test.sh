#!/usr/bin/env bash
set -x
DATAPATH="/data/scene_flow/"
CUDA_VISIBLE_DEVICES=2,3 python main_test.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 2 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/sceneflow/uniform_sample_d256  \
    --loadckpt "/home/zjlab/stereo_match/ORIGIN/CFNet/checkpoints/sceneflow/uniform_sample_d256/checkpoint_000019.ckpt" \
    --test_batch_size 2
    #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/sceneflow_pretraining.ckpt" \
    #--loadckpt "/home/zjlab/stereo_match/ORIGIN/CFNet/checkpoints/sceneflow/uniform_sample_d256_mish/checkpoint_000024.ckpt" \
    #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/checkpoint_000099.ckpt" \
