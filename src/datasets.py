"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function, division

import collections
import glob
import os
import random
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils
import re
class Pad(object):
    """Pad image and mask to the desired size

    Args:
      size (int) : minimum length/width
      img_val (array) : image padding value
      msk_val (int) : mask padding value

    """
    def __init__(self, size, img_val, msk_val):
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1)// 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1)// 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        image = np.stack([np.pad(image[:,:,c], pad,
                         mode='constant',
                         constant_values=self.img_val[c]) for c in range(3)], axis=2)
        mask = np.pad(mask, pad, mode='constant', constant_values=self.msk_val)
        return {'image': image, 'mask': mask}
        #return {'image': image, 'mask': mask,'img_2':img_2}
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top: top + new_h,
                        left: left + new_w]
        mask = mask[top: top + new_h,
                    left: left + new_w]
        return {'image': image, 'mask': mask}
        #return {'image': image, 'mask': mask,'img_2':img_2}
class ResizeShorterScale(object):
    """Resize shorter side to a given value and randomly scale."""

    def __init__(self, shorter_side, low_scale, high_scale):
        assert isinstance(shorter_side, int)
        self.shorter_side = shorter_side
        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        min_side = min(image.shape[:2])
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if min_side * scale < self.shorter_side:
            scale = (self.shorter_side * 1. / min_side)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask' : mask}
        #return {'image': image, 'mask': mask,'img_2':img_2}
class RandomMirror(object):
    """Randomly flip the image and the mask"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        do_mirror = np.random.randint(3)
        do_mirror=do_mirror-1
        image = cv2.flip(image, do_mirror)
        mask = cv2.flip(mask, do_mirror)
        return {'image': image, 'mask' : mask}
        #return {'image': image, 'mask': mask,'img_2':img_2}
class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        #img_2=sample['img_2]
        return {'image': (self.scale * image - self.mean) / self.std, 'mask' : sample['mask']}
        #return {'image': (self.scale * image - self.mean) / self.std, 'mask' : sample['mask'],'img_2':(self.scale * image - self.mean) / self.std)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}
        '''
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)
                'img_2':torch.from_numpy(img_2)}
        '''





#双输入时的数据处理

class Pad_double(object):
    """Pad image and mask to the desired size

    Args:
      size (int) : minimum length/width
      img_val (array) : image padding value
      msk_val (int) : mask padding value

    """
    def __init__(self, size, img_val, msk_val):
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1)// 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1)// 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        image = np.stack([np.pad(image[:,:,c], pad,
                         mode='constant',
                         constant_values=self.img_val[c]) for c in range(3)], axis=2)
        mask = np.pad(mask, pad, mode='constant', constant_values=self.msk_val)
        #return {'image': image, 'mask': mask}
        return {'image': image, 'mask': mask,'img_2':img_2}
class RandomCrop_double(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top: top + new_h,
                        left: left + new_w]
        mask = mask[top: top + new_h,
                    left: left + new_w]
        #return {'image': image, 'mask': mask}
        return {'image': image, 'mask': mask,'img_2':img_2}
class ResizeShorterScale_double(object):
    """Resize shorter side to a given value and randomly scale."""

    def __init__(self, shorter_side, low_scale, high_scale):
        assert isinstance(shorter_side, int)
        self.shorter_side = shorter_side
        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        min_side = min(image.shape[:2])
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if min_side * scale < self.shorter_side:
            scale = (self.shorter_side * 1. / min_side)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        img_2=cv2.resize(img_2, (500,500), interpolation=cv2.INTER_CUBIC)
        #return {'image': image, 'mask' : mask}
        return {'image': image, 'mask': mask,'img_2':img_2}
class RandomMirror_double(object):
    """Randomly flip the image and the mask"""

    def __init__(self):
        pass

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        do_mirror = np.random.randint(3)
        do_mirror=do_mirror-1
        image = cv2.flip(image, do_mirror)
        img_2=cv2.flip(img_2,do_mirror)
        mask = cv2.flip(mask, do_mirror)
        #return {'image': image, 'mask' : mask}
        return {'image': image, 'mask': mask,'img_2':img_2}
class Normalise_double(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        img_2=sample['img_2']
        #return {'image': (self.scale * image - self.mean) / self.std, 'mask' : sample['mask']}
        return {'image': (self.scale * image - self.mean) / self.std, 'mask': sample['mask'],'img_2':(self.scale * img_2 - self.mean) / self.std}

class resize_double(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        image=cv2.resize(image, (500,500), interpolation=cv2.INTER_CUBIC)
        mask=cv2.resize(mask, (500,500), interpolation=cv2.INTER_CUBIC)
        img_2=cv2.resize(img_2, (500,500), interpolation=cv2.INTER_CUBIC)
        
        return {'image': image,
                'mask': mask,
                'img_2':img_2}
class one_hot_double(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        one_hot_mask=np.zeros((500,500,4))
        for i in range(500):
            for j in range(500):
                one_hot_mask[i][j][mask[i][j]]=1
        one_hot_mask = one_hot_mask.transpose((2, 0, 1))
        return {'image': image,
                'mask': one_hot_mask,
                'img_2':img_2}
class one_hot(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask = sample['image'], sample['mask']
        one_hot_mask=np.zeros((500,500,4))
        for i in range(500):
            for j in range(500):
                one_hot_mask[i][j][mask[i][j]]=1
        one_hot_mask = one_hot_mask.transpose((2, 0, 1))
        return {'image': image,
                'mask': one_hot_mask
                }
class ToTensor_double(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #image, mask = sample['image'], sample['mask']
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        img_2 = img_2.transpose((2, 0, 1))
        #mask=torch.from_numpy(mask)
        '''
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}
        '''
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask),
                'img_2':torch.from_numpy(img_2)}
        

class ICIARDataset(Dataset):
    """ICIAR2018"""
    #其中mask为0的数据有13744张
    def __init__(
        self, data_file, data_dir, transform_trn=None, transform_val=None
        ):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform_{trn, val} (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = list(map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist))
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        mask_dir=self.root_dir+'splited_xml_little_P/'
        mask_file=''.join(self.datalist[idx])
        mask_name = os.path.join(mask_dir, mask_file)
        image1_dir=self.root_dir+'splited_svs_little/'
        image1_file=mask_file.replace('anno','image')
        image1_name= os.path.join(image1_dir, image1_file)
        #print(image1_name)
        #print(mask_name)
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        img_1 = read_image(image1_name)
        #mask = np.array(Image.open(mask_name).convert('P'))
        mask = np.array(Image.open(mask_name))
        '''
        for i in range(1000):
            for j in range(1000):
                a=mask[i][j]
                if a==15:
                    mask[i][j]=1
                elif a==40:
                    mask[i][j]=2
                elif a==190:
                    mask[i][j]=3
                else:
                    mask[i][j]=0
        '''        
        #print(np.shape(mask))
        #print(np.shape(img_1))
        #print('mask=',np.max(mask))
        #print(mask_name)
        
        sample = {'image': img_1, 'mask': mask}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        #print(sample['image'].shape)
        return sample
class ICIAR_double_Dataset(Dataset):
    """ICIAR2018"""
    #其中mask为0的数据有13744张
    def __init__(
        self, data_file, data_dir, transform_trn=None, transform_val=None
        ):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform_{trn, val} (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = list(map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist))
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        mask_dir=self.root_dir+'splited_xml_little_P/'
        mask_file=''.join(self.datalist[idx])
        mask_name = os.path.join(mask_dir, mask_file)
        image1_dir=self.root_dir+'splited_svs_little/'
        image1_file=mask_file.replace('anno','image')
        image1_name= os.path.join(image1_dir, image1_file)
        image2_dir=self.root_dir+'splited_svs_resize/'
        #image1_file=''.join(image1_file)
        middile_file=re.findall('.*image(.*).png.*',image1_file)
        middile_file=''.join(middile_file)
        image2_file=image1_file.replace(middile_file+'.png','.png')
        image2_name=os.path.join(image2_dir, image2_file)
        #print(image1_name)
        #print(image2_name)
        #print(mask_name)
        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        img_1 = read_image(image1_name)
        img_2=read_image(image2_name)
        #img_2=np.reshape(img_2,(500,500,3))
        #print(img_2.shape)
        #img_2=cv2.resize(img_2, (500,500), interpolation=cv2.INTER_CUBIC)
        #print(img_2.shape)
        mask = np.array(Image.open(mask_name))
        
        sample = {'image': img_1, 'mask': mask,'img_2':img_1}
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        #img_2=self.transform_val(img_2)
        #sample.update(image2=img_2)
        #print(sample['img_2'].shape)
        #print(sample['image'].shape)
        return sample
