import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from . import flow_transforms
import torchvision
import torchvision.transforms as transforms
import cv2
import copy

class KITTIDatasetCompletion(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.sparse_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        sparse_images = [x[2] for x in splits]
        disp_images = [x[3] for x in splits]
        return left_images, right_images, sparse_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        w,h = data.size
        baseline = 0.54
        width_to_focal = dict()
        width_to_focal[1242] = 721.5377
        width_to_focal[1241] = 718.856
        width_to_focal[1224] = 707.0493
        width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
        width_to_focal[1238] = 718.3351
        data = np.array(data, dtype=np.float32) / 256.
        
        conversion_rate = width_to_focal[w]*baseline
        data[data>0.01] = conversion_rate / (data[data>0.01])
        data[data<0.01] = 0
        #data[data>0] = 386.0 / (data[data>0])
        return data, conversion_rate

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
        

        if self.training:
            left_img = self.load_image(os.path.join(self.datapath, 'train/' + self.left_filenames[index]))
            right_img = self.load_image(os.path.join(self.datapath, 'train/' +self.right_filenames[index]))
            sparse, conversion_rate = self.load_disp(os.path.join(self.datapath, 'train/' +self.sparse_filenames[index]))
            disparity, conversion_rate = self.load_disp(os.path.join(self.datapath, 'train/' +self.disp_filenames[index]))
        else:
            left_img = self.load_image(os.path.join(self.datapath, 'val/' + self.left_filenames[index]))
            right_img = self.load_image(os.path.join(self.datapath, 'val/' + self.right_filenames[index]))
            sparse, conversion_rate = self.load_disp(os.path.join(self.datapath, 'val/' + self.sparse_filenames[index]))
            disparity, conversion_rate = self.load_disp(os.path.join(self.datapath, 'val/' + self.disp_filenames[index]))
        
        w, h = left_img.size
        crop_w, crop_h = 1216, 256
        left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
        right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
        sparse = sparse[h - crop_h:h, w - crop_w: w]
        disparity = disparity[h - crop_h:h, w - crop_w: w]

        if self.training:
            th, tw = 256, 512
            #th, tw = 320, 704
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
            angle = 0;
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
            augmented, sparse, disparity = co_transform([left_img, right_img], sparse, disparity)
            left_img = augmented[0]
            right_img = augmented[1]
            right_img = np.require(right_img, dtype='u1', requirements=['O', 'W'])

            right_img.flags.writeable = True
            if np.random.binomial(1,0.2):
              sx = int(np.random.uniform(35,100))
              sy = int(np.random.uniform(25,75))
              cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
              cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
              right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            sparse = np.ascontiguousarray(sparse, dtype=np.float32)
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            # get sparse input 
            #randmask = (np.random.randint(0, 100, size=(th, tw)) < 26) & (disparity > 0.1)
            randmask =  sparse > 0.1
            mask_val = np.zeros((th,tw), dtype=int)
            mask_val[randmask] = 1
            sparse_disparity= sparse*mask_val
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
                    "conversion_rate": conversion_rate}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            '''
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            
            assert len(disparity.shape) == 2
            sparse = np.lib.pad(sparse, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            #randmask = (np.random.randint(0, 100, size=(h+top_pad, w+right_pad)) < 26) & (disparity > 0.1)
            '''
            top_pad = 0
            right_pad = 0
            #randmask =  (sparse > 0.1)
            randmask = (np.random.randint(0, 100, size=(h+top_pad, w+right_pad)) < 100) & (sparse > 0.1)
            mask_val = np.zeros((h+top_pad, w+right_pad), dtype=int)
            mask_val[randmask] = 1
            sparse_disparity= sparse*mask_val
            sparse_disparity = transforms.ToTensor()(sparse_disparity)
            sparse_mask = transforms.ToTensor()(mask_val)

            #mask = sparse < 36
            #disparity[mask] = 0
            return {"left": left_img,
                    "right": right_img,
                    "sparse": sparse_disparity.float(),
                    "sparse_mask": sparse_mask.int(),
                    "disparity": disparity,
                    "conversion_rate": conversion_rate,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index]
                    }
            
