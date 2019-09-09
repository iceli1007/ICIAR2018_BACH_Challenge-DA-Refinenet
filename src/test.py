import argparse
import logging
import os
import random
import re
import sys
import time

# misc
import cv2
import numpy as np

# pytorch libs
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# custom libs
from config import *
import pyximport
from PIL import Image
pyximport.install()
from matplotlib import pyplot as plot
from skimage import io,transform
from miou_utils import compute_iu, fast_cm
from util import *
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
    # Custom libraries 
import sys
sys.path.append('..') 
from models.resnet import RefineNet_CA,Bottleneck,BasicBlock,ResNet152LW_double,RefineNet,Refinenet_double,Refinenet_double_concat

#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#gpu_ids = [5]
PATH='./ckpt/checkpoint_refinenet101_50_double.pth.tar'
scale=1./255, # SCALE
mean=np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)) # MEAN
std=np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)) # STD
test_dir='./data/'
test_file='./data/train_A05.txt'
model = Refinenet_double(Bottleneck, [3, 8, 36, 3], num_classes=4)
#model= RefineNet_CA(Bottleneck, [3, 8, 36, 3], num_classes=4)
#model = nn.DataParallel(model)  
#model.eval()
#model.load_state_dict(torch.load(PATH))
checkpoint=torch.load(PATH)
#checkpoint=torch.load(PATH, map_location=lambda storage, loc: storage.cuda(6))
#checkpoint=torch.load(PATH, map_location=torch.device('cpu'))
#print(checkpoint)
#print(checkpoint['segmenter'])
state_dict={k[7:]: v for k, v in checkpoint['segmenter'].items()}
#print(state_dict)
model.load_state_dict(state_dict)
#params = list(model.named_parameters())
#params=list(params)
#print(params[1])


model.cuda()
#model.to(device)
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

    def __call__(self, img):
        return (self.scale * img - self.mean) / self.std

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return  torch.from_numpy(image)
class Resize(object):
    """Resize the image to definite size"""
    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, image):
        image2= cv2.resize(image, None, fx=self.size, fy=self.size, interpolation=cv2.INTER_CUBIC)
        return image          
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
        #img_2=sample['img_2]
        return {'image': (self.scale * image - self.mean) / self.std, 'img_2':(self.scale * img_2 - self.mean) / self.std}

class ToTensor_double(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image,img_2 = sample['image'],sample['img_2']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        img_2=img_2.transpose((2,0,1))
        #mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                 'img_2':torch.from_numpy(img_2)
                }
class ICIAR_test_Dataset(Dataset):
    """ICIAR2018"""
    #其中mask为0的数据有13744张
    def __init__(
        self, data_file, data_dir, transform_test=None
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
        self.transform_test = transform_test
        self.stage = 'test'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_dir=self.root_dir+'train_A05_image/'
        img_file=''.join(self.datalist[idx])
        img_name = os.path.join(img_dir, img_file)
        image2_dir='./data/splited_svs_resize/'
        #image1_file=''.join(image1_file)
        middile_file=re.findall('.*image(.*).png.*',img_file)
        middile_file=''.join(middile_file)
        image2_file=img_file.replace(middile_file+'.png','.png')
        image2_name=os.path.join(image2_dir, image2_file)
        def read_image(x):
            img_arr = np.array(Image.open(x).resize((500,500)))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        img= read_image(img_name)
        img_2=read_image(image2_name)
        #print(np.shape(img))
        #io.imshow(img)
        #plt.show()
        sample_1={'image':img,'img_2':img_2}
        sample_1=self.transform_test(sample_1)
        img=sample_1['image']
        img_2=sample_1['img_2']
        #sample={'image':sample_1,'path':img_file}
        sample={'image':img,'path':img_file,'img_2':img_2}
        return sample
composed_test = transforms.Compose([Normalise_double(scale,mean,std),
                                
                                ToTensor_double()])
test_data=ICIAR_test_Dataset(data_file=test_file,
                data_dir=test_dir,
                transform_test=composed_test)
test_dataloader= DataLoader(test_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)
files=[]
masks=[]
img_all=np.zeros((40000,60000,3),dtype=int).astype(np.uint8)

#img_all=np.zeros((45000,50000,3),dtype=int).astype(np.uint8)

#img_all=np.zeros((40000,55000,3),dtype=int).astype(np.uint8)



with torch.no_grad():
    for w,sample in enumerate(test_dataloader):
        input=sample['image']
        path=sample['path']
        img_2=sample['img_2']
        input_var = torch.autograd.Variable(input).float().cuda()
        input_var_2 = torch.autograd.Variable(img_2).float().cuda()
        mask=model(input_var,input_var_2)
        #mask = nn.LogSoftmax()(mask)
        #mask=mask[0, :4].data.cpu().numpy().transpose(1, 2, 0)

        mask = cv2.resize(mask[0, :4].data.cpu().numpy().transpose(1, 2, 0),
                             (1000,1000),
                             interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
        
        
        mask_finally=np.zeros((1000,1000,3))
        
        for i in range(1000):
            for j in range(1000):
                if mask[i,j] !=0:
                    mask_finally[i,j,mask[i,j]-1]=255
        #files.append(path)
        
        #masks.append(mask_finally)
        id = ''.join(path)
        id=str(id)
        ids=id.split('_')
        print(ids)
        a=int(ids[1])
        #print(a)
        b=int(ids[2])
        c=int(ids[4][:-4])
        d=c%5
        e=c//5
        #print(id)
        print(a,b,d,e)
        mask=np.array(mask)
        #print(np.shape(mask))
        #i=np.array(img_all[(a+e*1000):(a+e*1000+1000),b+d*1000:b+d*1000+1000,:])
    #print(np.shape(i))
    #print(a+e*1000)
    #print(a+e*1000+1000)
        img_all[a+e*1000:a+e*1000+1000,b+d*1000:b+d*1000+1000,:]=mask_finally[:,:,:]
print('finished')
print(img_all)
print(np.shape(img_all))
img_all=cv2.resize(img_all,(1958,1317), interpolation=cv2.INTER_CUBIC).astype(np.uint8)

#img_all=cv2.resize(img_all,(1612,1333), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
#img_all=cv2.resize(img_all,(1826,1249), interpolation=cv2.INTER_CUBIC).astype(np.uint8)




print(np.shape(img_all))
io.imshow(img_all)
plt.show()
