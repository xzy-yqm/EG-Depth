#!/usr/bin/env bash
set -x
#DATAPATH="/data/syn_data_set/test/zjlab_aitown"
#DATAPATH="/data/syn_data_set/data_set3"
#DATAPATH="/data/syn_data_set/test/data_set3"
#DATAPATH="/data/502_dataset/test/before_rectify"
DATAPATH="/mnt/nas/kitti_completion/"
CUDA_VISIBLE_DEVICES=0 python save_disp_zjlab_with_pointcloud.py \
        --datapath $DATAPATH \
        --testlist ./filenames/kitti_completion_val1.txt \
        --dataset  kitti_completion  \
        --model cfnet --maxdisp 256 \
        --loadckpt "./checkpoints/checkpoint_000016.ckpt"
        #--loadckpt "./checkpoints/sceneflow/sgm_sparse_feature_guided_modify_d512/sgm_sparse_feature_guided_modify_d512.ckpt"
        #--loadckpt "./checkpoints/sceneflow/sgm_sparse_feature_guided_d512/final.ckpt"
        #--loadckpt "./checkpoints/sceneflow/sgm_sparse_feature_guided/checkpoint_000019.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/test_ldar_sterao3/checkpoints/502_xidian/sgm_sparse_feature_guided/checkpoint_000399.ckpt"
        #--loadckpt "./checkpoints/sceneflow/sgm_sparse_feature_guided/checkpoint_000019.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/test_lidar_stereo_full/checkpoints/502_xidian/sgm_guided_only/checkpoint_000399.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/test_ldar_sterao3/checkpoints/502_xidian/sgm_sparse_feature_guided/checkpoint_000399.ckpt"
        #--loadckpt "/home/zjlab/stereo_match/test_lidar_stereo_full/checkpoints/502_xidian/sgm_guided_only/checkpoint_000399.ckpt"
        #--loadckpt "./checkpoints/sceneflow/sgm_sparse_feature_guided/checkpoint_000019.ckpt"
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
