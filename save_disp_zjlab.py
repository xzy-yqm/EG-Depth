from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
from models import __models__, model_loss
import cv2
import sys


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test(file_path):
    #os.makedirs('./predictions', exist_ok=True)
    print('start testing')
    for batch_idx, sample in enumerate(TestImgLoader):
        print('start testing in')
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        disp_est_gt = tensor2numpy(sample["disparity"])
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_gt, disp_est, top_pad, right_pad, fn in zip(disp_est_gt, disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            print(disp_est.shape)
            #disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            disp_est = np.array(disp_est, dtype=np.float32)
            #fn = os.path.join("predictions", fn.split('/')[-1])
            fn = os.path.join(file_path, fn.split('/')[-1])
            fn1 = os.path.join(file_path, 'gt_'+fn.split('/')[-1])
            fn2 = os.path.join(file_path, 'color_'+fn.split('/')[-1])
            fn3 = os.path.join(file_path, 'gt_color_'+fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            #disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            #disp_est_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_est, alpha=1.0),cv2.COLORMAP_JET)
            #disp_gt_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_gt, alpha=1.0),cv2.COLORMAP_JET)
            print(disp_est.max(), disp_est.min())
            disp_est = (disp_est - disp_est.min())/(disp_est.max() - disp_est.min())
            print('new', disp_est.max(), disp_est.min())
            disp_gt = (disp_gt - disp_gt.min())/(disp_gt.max()-disp_gt.min())
            #disp_est_uint = np.round(disp_est).astype(np.uint16)
            disp_est_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_est*256, alpha=1.0),cv2.COLORMAP_JET)
            disp_gt_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_gt*256, alpha=1.0),cv2.COLORMAP_JET)
            #skimage.io.imsave(fn, disp_est_uint)
            cv2.imwrite(fn2, disp_est_color)
            cv2.imwrite(fn3, disp_gt_color)
            skimage.io.imsave(fn, disp_est)
            skimage.io.imsave(fn1, disp_gt)

def test1(file_path):
    #os.makedirs('./predictions', exist_ok=True)
    print('start testing')
    for batch_idx, sample in enumerate(TestImgLoader):
        print('start testing in')
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample1(sample))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            print(disp_est.shape)
            #disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            disp_est = np.array(disp_est, dtype=np.float32)
            #fn = os.path.join("predictions", fn.split('/')[-1])
            fn = os.path.join(file_path, fn.split('/')[-1])
            fn1 = os.path.join(file_path, 'gt_'+fn.split('/')[-1])
            fn2 = os.path.join(file_path, 'color_'+fn.split('/')[-1])
            fn3 = os.path.join(file_path, 'gt_color_'+fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est = disp_est[384:-1,1:-1]
            print(disp_est.shape)
            #disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            print(disp_est.max(), disp_est.min())
            #disp_est_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_est, alpha=1.0),cv2.COLORMAP_JET)
            disp_est = (disp_est - disp_est.min())/(disp_est.max() - disp_est.min())
            print('new', disp_est.max(), disp_est.min())
            disp_est_color=cv2.applyColorMap(cv2.convertScaleAbs(disp_est*256, alpha=1.0),cv2.COLORMAP_JET)
            #disp_est_uint = np.round(disp_est).astype(np.uint16)
            #skimage.io.imsave(fn, disp_est_uint)
            cv2.imwrite(fn2, disp_est_color)
            skimage.io.imsave(fn, disp_est)

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    print(sample['left'].shape)
    print(sample['right'].shape)
    disp_ests, pred3_s3, pred3_s4 = model(sample['left'].cuda(), sample['right'].cuda())
    disp_gt = sample['disparity'].cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)
    print(disp_gt)
    scalar_outputs = {"loss": loss}
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1_preds3"] = [D1_metric(pred, disp_gt, mask) for pred in pred3_s3]
    scalar_outputs["D1_preds4"] = [D1_metric(pred, disp_gt, mask) for pred in pred3_s4]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres1s3"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in pred3_s3]
    scalar_outputs["Thres1s4"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in pred3_s4]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres2s3"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in pred3_s3]
    scalar_outputs["Thres2s4"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in pred3_s4]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    print(disp_ests[-1].shape)
    print(disp_ests)
    print(scalar_outputs)
    return disp_ests[-1]


@make_nograd_func
def test_sample1(sample):
    model.eval()
    print(sample['left'].shape)
    print(sample['right'].shape)
    disp_ests, pred3_s3, pred3_s4 = model(sample['left'].cuda(), sample['right'].cuda())

    print(disp_ests[-1].shape)
    return disp_ests[-1]

if __name__ == '__main__':
    file_path = "/home/zjlab/stereo_match/CFNet_bugfix/pre_picture"
    #test(sys.argv[1])

    test1(file_path)
