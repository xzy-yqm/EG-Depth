import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from . import flow_transforms
import torchvision
import cv2
import copy
import torchvision.transforms as transforms

class ZjlabDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.disp_filenames_gt = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        disp_images_gt = [x[3] for x in splits]
        return left_images, right_images, disp_images, disp_images_gt

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename, mode):
        #data, scale = pfm_imread(filename)
        #data = np.ascontiguousarray(data, dtype=np.float32)
        #cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #return data
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #image = 0.26*3672/(np.ascontiguousarray(image[:,:,1],dtype=np.float32))
        #image = 0.27*1739/(np.ascontiguousarray(image[:,:,1],dtype=np.float32))
        # depend on 5.5um, 12mm focal lenth, 0.27mm distance
        #print(filename, image)
        if mode == 0:
            image[image>0] = 589.09 / (image[image>0]/256)
        else:
            image[image>0] = 589.09 / (image[image>0])
        #image = np.ascontiguousarray(image[:,:],dtype=np.float32)
        #print(filename, image.size,image.shape)
        return image

    # def RGB2GRAY(self, img):
    #     imgG = copy.deepcopy(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     imgG[:, :, 0] = img
    #     imgG[:, :, 1] = img
    #     imgG[:, :, 2] = img
    #     return imgG

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity_sparse = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]), 0)
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames_gt[index]), 1)
        # left_img = self.RGB2GRAY(left_img)
        # right_img = self.RGB2GRAY(right_img)
        #print(left_img.size, disparity_sparse.size, disparity.size)
        if self.training:

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

            # w, h  = left_img.size
            # th, tw = 256, 512
            #
            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)
            #
            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # right_img = np.asarray(right_img)
            # left_img = np.asarray(left_img)

            # geometric unsymmetric-augmentation
            angle = 0
            px = 0
            if np.random.binomial(1, 0.5):
                # angle = 0.1;
                # px = 2
                angle = 0.05
                px = 1
            co_transform = flow_transforms.Compose2([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop2((th, tw)),
            ])
            augmented, disparity, disparity_sparse = co_transform([left_img, right_img], disparity, disparity_sparse)
            left_img = augmented[0]
            right_img = augmented[1]

            # randomly occlude a region
            
            right_img = np.require(right_img, dtype='u1', requirements=['O', 'W'])
            right_img.flags.writeable = True
            if np.random.binomial(1,0.5):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # w, h = left_img.size
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            disparity_sparse = np.ascontiguousarray(disparity_sparse, dtype=np.float32)
            # get sparse input 
            randmask = (disparity_sparse > 0.1)
            #randmask = (np.random.randint(0, 100, size=(th, tw)) < 10) & (disparity > 0.1)
            mask_val = np.zeros((th,tw), dtype=int)
            mask_val[randmask] = 1
            #sparse_disparity= disparity*mask_val
            sparse_disparity= disparity_sparse*mask_val
            sparse_disparity = transforms.ToTensor()(sparse_disparity)
            sparse_mask = transforms.ToTensor()(mask_val)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)



            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "sparse": sparse_disparity.float(),
                    "sparse_mask": sparse_mask.int()}
        else:
            w, h = left_img.size
            #crop_w, crop_h = 2048, 2048
            crop_w, crop_h = 2048, 1024
            #crop_w, crop_h = 2048, 1024
            #w = 2248
            #h = 1576

            #left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            #right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            #disparity = disparity[h - crop_h:h, w - crop_w: w]
            #disparity_sparse = disparity_sparse[h - crop_h:h, w - crop_w: w]
            left_img = left_img.crop((0, 0, crop_w, crop_h))
            right_img = right_img.crop((0, 0, crop_w, crop_h))
            disparity = disparity[0:crop_h, 0:crop_w]
            disparity_sparse = disparity_sparse[0:crop_h, 0:crop_w]
            #left_img = left_img.crop(((w - crop_w)//2, (h - crop_h)//2, (w + crop_w)//2, (h + crop_h)//2))
            #right_img = right_img.crop(((w - crop_w)//2, (h - crop_h)//2, (w + crop_w)//2, (h + crop_h)//2))
            #disparity = disparity[(h - crop_h)//2:(h + crop_h)//2, (w - crop_w)//2:(w + crop_w)//2]
            #disparity_sparse = disparity_sparse[(h - crop_h)//2:(h + crop_h)//2, (w - crop_w)//2:(w + crop_w)//2]

            disparity_sparse = np.ascontiguousarray(disparity_sparse, dtype=np.float32)
            # get sparse input 
            randmask = (disparity_sparse > 0.1)
            #randmask = (disparity > 0.1)
            #randmask = (np.random.randint(0, 100, size=(crop_h, crop_w)) < 10) & (disparity > 0.1)
            mask_val = np.zeros((crop_h,crop_w), dtype=int)
            mask_val[randmask] = 1
            #sparse_disparity= disparity*mask_val
            sparse_disparity= disparity_sparse*mask_val
            sparse_disparity = transforms.ToTensor()(sparse_disparity)
            sparse_mask = transforms.ToTensor()(mask_val)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "sparse": sparse_disparity.float(),
                    "sparse_mask": sparse_mask.int(),
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index]}
