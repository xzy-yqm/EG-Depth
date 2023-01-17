import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision
import cv2
import copy
import sys
import os
import re

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    if data.size != height*width:
        print(filename)
        print(data.size, height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def cal_dataset_distribute(datapath, list_filename, output_file):
    ''' calculate the disparity distribution of the target dataset '''
    lines = read_all_lines(list_filename)
    splits = [line.split() for line in lines]
    disp_images = [x[2] for x in splits]
    disp_images = disp_images[29900:29999]
    print(disp_images[0],disp_images[-1])
    output = np.zeros((1,50000))
    for index in range(0, len(disp_images)):
        data, scale = pfm_imread(os.path.join(datapath, disp_images[index]))
        w,h = data.shape
        if w != 540 or h != 960:
            print(disp_images[index])
            print(w,h)
            return
        for x in range(1, w):
            for y in range(1,h):
                output[0][int(data[x, y]*100)] = output[0][int(data[x, y]*100)] + 1
    
    fd = open(output_file, 'w')
    for i in range(1, 50000):
        fd.write("%ld ,\n" %(output[0][i]))
    fd.close()


if __name__ == "__main__":
    print(sys.argv)
    cal_dataset_distribute(sys.argv[1], sys.argv[2], sys.argv[3])
