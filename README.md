# EG-net

This is the implementation of paper "Expanding Sparse LiDAR Depth and Guiding Stereo Matching for Robust Dense Depth Estimation" on RAL 2023.

Camera ready version and meterials will be ginven.


## Abstract
Dense depth estimation is an important task for various applications, such as object detection, 3-D reconstruction, etc. Stereo matching, as a popular method for dense depth estimation, is faced with challenges when low textures, occlusions or domain gaps exist. Stereo-LiDAR fusion has recently become a promising way to deal with these challenges. However, due to the sparsity and uneven distribution of the LiDAR depth data, existing stereo-LiDAR fusion methods tend to ignore the data when their density is quite low or they largely differ from the depth predicted from stereo images. To provide a solution to this problem, we propose a stereo-LiDAR fusion method by first expanding the sparse LiDAR depth to semi-dense depth with RGB image as reference. Based on the semi-dense depth, a varying-weight Gaussian guiding method is proposed to deal with the varying reliability of guiding signals. A multi-scale feature extraction and fusion method is further used to enhance the network, which shows superior performance than traditional sparse invariant convolution methods. Experimental results on different public datasets demonstrate our superior accuracy and robustness over the state-of-the-arts.


# How to use

## Environment
* python                    3.6.2
* pytorch                   1.10.1 
* numpy                     1.19.2


## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), 
[KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), 
[KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), 
[KITTI COMPLETION](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion),
[Middlebury](https://vision.middlebury.edu/stereo/)

**KITTI2015/2012 SceneFlow**

please place the dataset as described in `"./filenames"`, i.e., `"./filenames/sceneflow_train.txt"`, `"./filenames/sceneflow_test.txt"`, `"./filenames/kitti_completion_train.txt"`, `"./filenames/kitti_completion_test.txt"`


## Training
**Scene Flow Datasets Pretraining**

run the script `./scripts/sceneflow.sh` to pre-train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.

To repeat our pretraining details. 

**Kitti completion Datasets Pretraining**

run the script `./scripts/kitti_completion.sh` to pre-train on KITTI completion datsets. Please update `DATAPATH` in the bash file as your training data path.

To repeat our pretraining details. 

## Evaluation
run the script `./scripts/kitti_completion_test.sh` to evaluate the performance on KITTI completion datsets.

## Pretrained Models
[Pretraining Model]()
You can use this checkpoint to reproduce the result we reported in Table.I of the main paper.

## Citation
If you find this code useful in your research, please cite:
```
@artical {IEEE ROBOTICS AND AUTOMATION LETTERS,
    author    = {Zhenyu Xu , Yuehua Li, Shiqiang Zhu, and Yuxiang Sun},
    title     = {Expanding Sparse LiDAR Depth and Guiding Stereo Matching for Robust Dense Depth Estimation},
    booktitle = {},
    month     = {},
    year      = {2023},
    pages     = {}
}
```
# Acknowledgements

Thanks to the excellent work CF-NET, GWCNet, Deeppruner, and HSMNet. Our work is inspired by these work and part of codes are migrated from [CF-NET](https://github.com/gallenszl/CFNet), [GWCNet](https://github.com/xy-guo/GwcNet), [DeepPruner](https://github.com/uber-research/DeepPruner/) and [HSMNet](https://github.com/gengshan-y/high-res-stereo).