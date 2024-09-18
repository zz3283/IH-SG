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



class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split
        self.image_paths=dataroot
        self.gt_path, self.mask_path, self.input_path = [], [], []

        if self.split== 'train':
                print('loading training file...')
                self.trainfile = os.path.join(self.image_paths, 'IHD_train.txt')
                with open(self.trainfile, 'r') as f:
                    for line in f.readlines():
                        line = line.rstrip()
                        name_parts = line.split('_')
                        data_parts = line.split('/')
                        # if data_parts[0] != 'HAdobe5k':
                        #    continue
                        mask_path = line.replace('composite_images', 'masks')
                        mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')

                        gt_path = line.replace('composite_images', 'real_images')
                        gt_path = gt_path.replace('_' + name_parts[-2] + '_' + name_parts[-1], '.jpg')

                        self.gt_path.append(os.path.join(self.image_paths, gt_path))
                        self.input_path.append(os.path.join(self.image_paths, line))
                        self.mask_path.append(os.path.join(self.image_paths, mask_path))


                self.dataset_len = len(self.gt_path)
                if self.data_len <= 0:
                        self.data_len = self.dataset_len
                else:
                        self.data_len = min(self.data_len, self.dataset_len)

        elif self.split=='val':
                print('loading val file...')
                self.trainfile = os.path.join(self.image_paths, 'IHD_test.txt')
                with open(self.trainfile, 'r') as f:
                    for line in f.readlines():
                        line = line.rstrip()
                        name_parts = line.split('_')
                        data_parts = line.split('/')
                        # if data_parts[0] != 'HAdobe5k':
                        #    continue
                        mask_path = line.replace('composite_images', 'masks')
                        mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')

                        gt_path = line.replace('composite_images', 'real_images')
                        gt_path = gt_path.replace('_' + name_parts[-2] + '_' + name_parts[-1], '.jpg')

                        self.gt_path.append(os.path.join(self.image_paths, gt_path))  # 有阴影  /255
                        self.input_path.append(os.path.join(self.image_paths, line))  # 无阴影
                        self.mask_path.append(os.path.join(self.image_paths, mask_path))
                        #前景的mask 初始化阴影的mask可以取1-mask

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_gt = None
        img_input = None
        img_mask = None
        mask_gt = None
        mask_start=None
        shadow_mask=None ##阴影区域的初始mask  -- 1-img_mask 替代？？？不知道有没有用

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')    # r_res --> resolution
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
        else:

            img_gt = torch.from_numpy(load_img_crop(self.gt_path[index]))
            img_input = torch.from_numpy(load_img_crop(self.input_path[index]))
            img_mask = torch.from_numpy(load_img_crop_gray(self.mask_path[index]))

            img_mask = img_mask.unsqueeze(-1)  #[b w h c]---[w h c]
            img_gt = img_gt.permute(2,0,1)
            img_input = img_input.permute(2,0,1)
            img_mask = img_mask.permute(2,0,1)
            # print(img_gt.shape) [3,356,256]


            # ##----##用于阴影的mask初始不存在--所以初始用前景mask 的逆mask 替代
            # shadow_mask=1-img_mask
            # shadow_mask = torch.where(shadow_mask > 0.05, 1., 0.)  ##好像需要调比例 要不然不对
            # mask_start=shadow_mask #img_mask
            # #阴影的mask的Gt  shadow-noshadow
            # img_mask_gt = (img_input - img_gt)
            # mask_gt = img_mask_gt[0] * 0.299 + img_mask_gt[1] * 0.387 + img_mask_gt[2] * 0.114
            # mask_gt = mask_gt.unsqueeze(-1)    ## [256,256,1]  h w c
            # mask_gt = mask_gt.permute(2, 0, 1) #  [1,256,256] c h w
            # mask_gt = torch.where(mask_gt>0.05, 1., 0.)

        ##for harmony
        [ img_gt, img_input] = Util.transform_augment(
            [img_gt, img_input], split=self.split, min_max=(0, 1))
        [img_mask,] = Util.transform_augment([img_mask], split=self.split, min_max=(0, 1))
        return {'GT': img_gt, 'Input': img_input,'Mask': img_mask,'Index': index}
        #---------
        #Mask 是前景mask


        # return {'Degra':img_degra, 'GT': img_gt, 'Input': img_input, 'Mask': img_mask, 'Mask_GT':img_mask_gt, 'Index': index}
        # [ img_gt, img_input] = Util.transform_augment(
        #     [img_gt, img_input], split=self.split, min_max=(0, 1))
        # [img_mask, mask_gt,shadow_mask] = Util.transform_augment([img_mask, mask_gt,shadow_mask], split=self.split, min_max=(0, 1))
        # return {'GT': img_gt, 'Input': img_input,'Mask_start':mask_start,
        #         'Mask_GT': mask_gt, 'Mask': img_mask, 'shadow_mask':shadow_mask,'Index': index}
        # # return {'Degra':img_degra, 'GT': img_gt, 'Input': img_input, 'Mask': img_mask, 'Mask_GT':img_mask_gt, 'Index': index}


