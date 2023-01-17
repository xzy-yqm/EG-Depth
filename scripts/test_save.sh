#!/usr/bin/env bash
set -x
DATAPATH="/data/syn_data_set/test/zjlab_aitown"
#DATAPATH="/data/syn_data_set/data_set3"
#DATAPATH="/data/syn_data_set/test/data_set3"
CUDA_VISIBLE_DEVICES=0 python save_disp_zjlab.py \
        --datapath $DATAPATH \
        --testlist ./filenames/zjlab_test_aitown3.txt \
        --dataset  zjlab  \
        --model cfnet --maxdisp 576 \
        --loadckpt "/home/zjlab/stereo_match/CFNet_bugfix/checkpoints/sceneflow/maxdisp_576/checkpoint_000019.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/sceneflow_pretraining.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/zjlab_dataset10_bugfix/checkpoint_000399.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/sceneflow_pretraining.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/zjlab_dataset10_bugfix/checkpoint_000399.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/sceneflow_pretraining.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/CFNet_bakup/checkpoints/zjlab_abstudy_test3/checkpoint_000060.ckpt"
        #--dataset  kitti  \
        #--testlist ./filenames/zjlab_test_dataset10.txt \
        #--testlist ./filenames/zjlab_test_kitti.txt \
        #--testlist ./filenames/zjlab_test_502.txt \
        #--testlist ./filenames/zjlab_test.txt \
        #--testlist ./filenames/zjlab_test_dataset3.txt \
        #--dataset  sceneflow  \
