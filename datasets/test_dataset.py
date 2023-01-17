import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data_io import get_transform, read_all_lines, pfm_imread
import flow_transforms
import torchvision
import cv2
import copy
import sys
import torchvision.transforms as transforms


# test script: python3 test_dataset.py /data/syn_data_set/data_set10/  left_image/Image0339_L.png  right_image/Image0339_R.png disp_exr/Image0339_L.exr
# test script for sceneflow: python3 test_dataset.py /data/scene_flow/ FlyingThings3D_release/frames_cleanpass/TEST/B/0049/left/0010.png FlyingThings3D_release/frames_cleanpass/TEST/B/0049/right/0010.png FlyingThings3D_release/disparity/TEST/B/0049/left/0010.pfm
# test script for kitti: python3 test_dataset.py /data/kitti_2012/training/ colored_0/000193_10.png  colored_0/000193_11.png disp_occ/000193_10.png
# test script for middlebury: python3 test_dataset.py /data/middlebury/2014/Adirondack-perfect/ im0.png  im1.png disp0.pfm
# test script for 502 dataset:  python3 test_dataset.py /data/502_dataset/dataset/  left_rgb/000000.png right_rgb/000000.png left_depth/000000.tiff
def load_disp( filename):
    # for scene flow and middlebury
    #data, scale = pfm_imread(filename)
    #data = np.ascontiguousarray(data, dtype=np.float32)
    #cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #return data
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    #image = 0.26*3672/(np.ascontiguousarray(image[:,:,1],dtype=np.float32))
    print(image[image>0])
    print(image.max())
    image[image>0] = 645.3 / (image[image>0]/256)
    print(image.size, image.shape)
    return image
    # for kitti data set
    #data = Image.open(filename)
    #data = np.array(data, dtype=np.float32) / 256.
    #return data

def load_image(filename):
    return Image.open(filename).convert('RGB')

def test_data_set(datapath, left_filename, right_filename, disp_filename):
    ''' test the data set loaded'''
    left_img =  load_image(os.path.join(datapath, left_filename))
    right_img = load_image(os.path.join(datapath, right_filename))
    disparity = load_disp(os.path.join(datapath, disp_filename))
    print(disparity.size, left_img.size)
    w, h = left_img.size
    #crop_w, crop_h = 2048, 2048
    print('img width: ', w, 'img height: ', h)
    #crop_w, crop_h = 1238, 374
    crop_w, crop_h = 2048, 2048
    #crop_w, crop_h = 2880, 1988
    im_color1=cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=1.0),cv2.COLORMAP_JET)
    cv2.imwrite('disparity2_1.png', disparity)
    cv2.imwrite('disparity2_color1.png', im_color1)

    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    disparity = disparity[h - crop_h:h, w - crop_w: w]

    # calculate the mask
    mask_w = crop_w
    mask_h = crop_h

    randmask = (np.random.randint(0, 100, size=(mask_h, mask_w)) < 5) & (disparity > 0.1) 
    # randmask = (np.random.randint(0, 100, size=(mask_h, mask_w)) < 26) & (disparity > 0.1)
    mask_val = np.zeros((mask_h,mask_w), dtype=int)
    mask_val[randmask] = 1
    randmask1 = (disparity > 0.1)
    mask_val1 = np.zeros((mask_h,mask_w), dtype=int)
    mask_val1[randmask1] = 1
    print(randmask)
    print(mask_val)
    print('origin percentage:', sum(sum(mask_val1))/(mask_h*mask_w), ', mask percentage: ', sum(sum(mask_val))/(mask_h*mask_w))

    #processed = get_transform()
    #left_img = processed(left_img)
    #right_img = processed(right_img)

    left_img.save('left_image2.png', 'PNG')
    right_img.save('right_img2.png', 'PNG')

    #cv2.imwrite('left_image.png', left_img)
    #cv2.imwrite('right_img.png', right_img)
    print(disparity)
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=1.0),cv2.COLORMAP_JET)
    cv2.imwrite('disparity2.png', disparity)
    cv2.imwrite('disparity2_color.png', im_color)
    disparity[(np.isinf(disparity))] = 0
    disparity[( np.isnan(disparity))] = 0
    print(np.isnan(disparity).any())
    print(np.isinf(disparity).any())
    disparity_mask = disparity*mask_val
    disparity_mask_color=cv2.applyColorMap(cv2.convertScaleAbs(disparity_mask, alpha=1.0),cv2.COLORMAP_JET)
    cv2.imwrite('disparity3.png', disparity_mask)
    cv2.imwrite('disparity3_color.png', disparity_mask_color)
    # test the downsample in torch
    disparity_mask1 = transforms.ToTensor()(disparity_mask)
    disparity1 = disparity_mask1[:,0:-1:8,0:-1:8]
    print(disparity_mask, disparity_mask.size)
    print(disparity_mask1, disparity_mask1.size())
    print(disparity1, disparity1.size())


def test_data_set1(datapath, left_filename, right_filename, disp_filename):
    ''' test the data set loaded'''
    left_img =  load_image(os.path.join(datapath, left_filename))
    right_img = load_image(os.path.join(datapath, right_filename))
    disparity = load_disp(os.path.join(datapath, disp_filename))

    th, tw = 256, 512
    #th, tw = 288, 512
    random_brightness = np.random.uniform(0.5, 2.0, 2)
    random_gamma = np.random.uniform(0.8, 1.2, 2)
    random_contrast = np.random.uniform(0.8, 1.2, 2)
    left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
    left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
    left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
    right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
    right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
    right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
    right_img = np.asarray(right_img)
    left_img = np.asarray(left_img)

    # geometric unsymmetric-augmentation
    angle = 0
    px = 0
    if np.random.binomial(1, 0.5):
        # angle = 0.1;
        # px = 2
        angle = 0.05
        px = 1
    co_transform = flow_transforms.Compose([
        # flow_transforms.RandomVdisp(angle, px),
        # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
        flow_transforms.RandomCrop((th, tw)),
    ])
    augmented, disparity = co_transform([left_img, right_img], disparity)
    left_img = augmented[0]
    right_img = augmented[1]

    # randomly occlude a region
    
    right_img = np.require(right_img, dtype='f4', requirements=['O', 'W'])
    right_img.flags.writeable = True
    if np.random.binomial(1,0.5):
        sx = int(np.random.uniform(35,100))
        sy = int(np.random.uniform(25,75))
        cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
        cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
        right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

    # w, h = left_img.size

    disparity = np.ascontiguousarray(disparity, dtype=np.float32)

    #left_img.save('left_image1.png', 'PNG')
    #right_img.save('right_img1.png', 'PNG')

    cv2.imwrite('left_image1.png', left_img)
    cv2.imwrite('right_img1.png', right_img)

    cv2.imwrite('disparity1.png', disparity)

if __name__ == '__main__':
    print(sys.argv)
    test_data_set(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
