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


class testdata(Dataset):
    def __init__(self, dataroot, datatype, split='val', data_len=-1):
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
        if self.split=='val':
            # for shadow
            self.gt_path = Util.get_paths_from_images(os.path.join(dataroot, 'shadow_img'))
            self.input_path = Util.get_paths_from_images(os.path.join(dataroot, 'shadowfree_img'))
            self.mask_path = Util.get_paths_from_images(os.path.join(dataroot, 'foreground_object_mask'))
            self.dataset_len = len(self.gt_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))


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



