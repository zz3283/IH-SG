import os.path
from io import BytesIO
import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import cv2
import numpy as np
from torchvision import transforms


def load_img_crop(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
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

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            # for specular
            # self.gt_path = Util.get_paths_from_images(os.path.join(dataroot, 'test_D'))     # diffuse
            # self.input_path = Util.get_paths_from_images(os.path.join(dataroot, 'test_A'))  # highlight
            # self.mask_path = Util.get_paths_from_images(os.path.join(dataroot, 'test_M'))   # mask
            # self.degra_path = Util.get_paths_from_images(os.path.join(dataroot, 'test_H_256_2')) # 退化图
            # self.spec_path = Util.get_paths_from_images(os.path.join(dataroot, 'test_S'))   # specular

            # for shadow
            self.gt_path = Util.get_paths_from_images(os.path.join(dataroot, 'real_images'))
            self.input_path = Util.get_paths_from_images(os.path.join(dataroot, 'composite_images'))
            self.mask_path = Util.get_paths_from_images(os.path.join(dataroot, 'masks'))
            # self.mask_gt_path = Util.get_paths_from_images(os.path.join(dataroot, 'mask'))      # mask gt
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
        img_degra = None
        mask_gt = None

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
            img_degra = torch.from_numpy(load_img_crop(self.degra_path[index]))
            img_gt = torch.from_numpy(load_img_crop(self.gt_path[index]))
            img_input = torch.from_numpy(load_img_crop(self.input_path[index]))
            # img_spec = torch.from_numpy(load_img_crop_gray(self.spec_path[index]))
            img_mask = torch.from_numpy(load_img_crop_gray(self.mask_path[index]))
            # img_spec = img_spec.unsqueeze(-1)
            img_mask = img_mask.unsqueeze(-1)
            img_degra = img_degra.permute(2,0,1)
            img_gt = img_gt.permute(2,0,1)
            img_input = img_input.permute(2,0,1)
            # img_spec = img_spec.permute(2,0,1)
            img_mask = img_mask.permute(2,0,1)
            # degra use gt
            # img_degra = img_input / (img_gt + 0.0001)
            img_mask_gt = img_input - img_gt
            mask_gt = img_mask_gt[0] * 0.299 + img_mask_gt[1] * 0.387 + img_mask_gt[2] * 0.114
            mask_gt = mask_gt.unsqueeze(-1)
            mask_gt = mask_gt.permute(2, 0, 1)
            mask_gt = torch.where(mask_gt>0.1, 1., 0.)

        ##for harmony
        [ img_gt, img_input] = Util.transform_augment(
            [img_gt, img_input], split=self.split, min_max=(0, 1))
        [img_mask,] = Util.transform_augment([img_mask], split=self.split, min_max=(0, 1))
        return {'GT': img_gt, 'Input': img_input,'Mask': img_mask,'Index': index}
        # return {'Degra':img_degra, 'GT': img_gt, 'Input': img_input, 'Mask': img_mask, 'Mask_GT':img_mask_gt, 'Index': index}
