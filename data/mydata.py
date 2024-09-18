import os.path
from io import BytesIO
import lmdb
import argparse
import sys
sys.path.append("../..")
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
import data.util as Util
import core.metrics as Metrics
from torchvision import transforms


def load_img_crop(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    ##[480,640 3]--[h,w,c] RGB
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_img_crop_gray(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    # img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img


class mydata(Dataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split
        self.image_paths=dataroot
        self.gt_path, self.mask_path, self.input_path = [], [], []
        """
        合成图像：IMG_2230_2.jpg 
        gt：IMG_2230.jpg
        mask:IMG_2230.png  #8位
        """
        if self.split == 'train':
            print('loading training file...')
            self.trainfile = os.path.join(self.image_paths, 'mydata_train.txt')
            with open(self.trainfile, 'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    data_parts = line.split('/')

                    mask_path = line.replace(('_' + name_parts[-1]), '.jpg')
                    gt_path = line.replace(('_' + name_parts[-1]), '.jpg')

                    self.gt_path.append(os.path.join(self.image_paths + '/real_images/', gt_path))
                    self.input_path.append(os.path.join(self.image_paths + '/composite_images/', line))
                    # self.mask_path.append(os.path.join(self.image_paths + '/masks/', mask_path))

            self.dataset_len = len(self.gt_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        elif self.split == 'val':
            print('loading val file...')
            self.trainfile = os.path.join(self.image_paths, 'mydata_test.txt')  #newdata_test.txt
            with open(self.trainfile, 'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    mask_path = line.replace(('_' + name_parts[-1]), '.jpg') #
                    # print(mask_path)
                    gt_path = line.replace(('_' + name_parts[-1]), '.jpg')
                    # print(gt_path)

                    self.gt_path.append(os.path.join(self.image_paths + '/real_images/', gt_path))
                    self.input_path.append(os.path.join(self.image_paths + '/composite_images/', line))
                    # self.mask_path.append(os.path.join(self.image_paths + '/masks/', mask_path))
                    # 前景的mask 初始化阴影的mask可以取1-mask



    def __len__(self):
        return self.data_len
    def __getitem__(self, index):
        img_gt = None
        img_input = None
        img_mask = None

        if self.datatype == 'img':
            img_gt = torch.from_numpy(load_img_crop(self.gt_path[index]))
            img_input = torch.from_numpy(load_img_crop(self.input_path[index]))
            img_mask = torch.from_numpy(load_img_crop_gray(self.mask_path[index]))

            img_mask = img_mask.unsqueeze(-1)  #[b w h c]---[w h c]
            img_gt = img_gt.permute(2,0,1)
            img_input = img_input.permute(2,0,1)
            img_mask = img_mask.permute(2,0,1)

        ##for harmony testdata
        [ img_gt, img_input] = Util.transform_augment(
            [img_gt, img_input], split=self.split, min_max=(0, 1))
        [img_mask] = Util.transform_augment([img_mask], split=self.split, min_max=(0, 1))
        return {'GT': img_gt, 'Input': img_input,'Mask': img_mask,'Index': index}


