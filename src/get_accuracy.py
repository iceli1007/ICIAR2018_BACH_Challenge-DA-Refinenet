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
from models.resnet import RefineNet_CA,Bottleneck,BasicBlock,Refinenet_double_concat,ResNet152LW_double,RefineNet,ResNetLW_double,ResNetLW,Refinenet_double,Refinenet_double_attention
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
PATH='/data4/lzq/data/ICIAR_2018/bishe_ICIAR/light-weight-refinenet/ckpt/152_duochidu.tar'
scale=1./255, # SCALE
mean=np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)) # MEAN
std=np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)) # STD
test_dir='/data4/nhj/datasets/ICIAR2018_BACH_Challenge/train/WSI/'
test_file='/data4/nhj/datasets/ICIAR2018_BACH_Challenge/train/WSI/accuracy_test.txt'
model = Refinenet_double(Bottleneck, [3, 8, 36, 3], num_classes=4)
#model= RefineNet_CA(Bottleneck, [3, 8, 36, 3], num_classes=4)
#model = nn.DataParallel(model)  s
#model.eval()
#model.load_state_dict(torch.load(PATH))
#checkpoint=torch.load(PATH,map_location=lambda storage, loc: storage)
checkpoint=torch.load(PATH)

#print(checkpoint['segmenter'])
state_dict={k[7:]: v for k, v in checkpoint['segmenter'].items()}
print(state_dict)
model.load_state_dict(state_dict)
#params = list(model.named_parameters())
#params=list(params)
#print(params[1])

#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")#第一行代码
#model.to(device)#第二行代码
model.eval()
model.cuda()
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
        return {'image': (self.scale * image - self.mean) / self.std, 'img_2':(self.scale * img_2 - self.mean) / self.std,'mask' : sample['mask']}

class ToTensor_double(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        #image, mask,img_2 = sample['image'], sample['mask'],sample['img_2']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        img_2=img_2.transpose((2,0,1))
        #mask = mask.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                 'img_2':torch.from_numpy(img_2),
                'mask': torch.from_numpy(mask)}
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
        self.stage = 'val'

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
            img_arr = np.array(Image.open(x).resize((500,500)))
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
        
        
        
        
        
        '''
        mask_dir=self.root_dir+'splited_xml_little_P/'
        mask_file=''.join(self.datalist[idx])
        mask_name = os.path.join(mask_dir, mask_file)
        image1_dir=self.root_dir+'splited_svs_little/'
        image1_file=mask_file.replace('anno','image')
        image1_name= os.path.join(image1_dir, image1_file)
        #print(image1_name)
        #print(mask_name)
        def read_image(x):
            img_arr = np.array(Image.open(x).resize((500,500)))
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        img_1 = read_image(image1_name)
        #mask = np.array(Image.open(mask_name).convert('P'))
        mask = np.array(Image.open(mask_name))  
        #print(np.shape(mask))
        #print(np.shape(img_1))
        #print('mask=',np.max(mask))
        #print(mask_name)
        
        sample = {'image': img_1, 'mask': mask}
        '''
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_test:
                sample = self.transform_test(sample)
        #print(sample['image'].shape)
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
def get_accuracy(_np_gtlist,_np_probls, need_all=False,precision=3):
        """
        正确率公式： AC = (TP+TN)/(TP+TN+FP+FN)
        # TP : True Positive , TN: True Negative, FP: False Positive, FN : False Negative,
        :return:预测结果和分割图一致的点的数目/全部的像素点数
        """
        all_points = np.prod(_np_gtlist[0].shape)
        #all_points=1000000
        acc_list = []
        #print(all_points)
        for gt, gtprob in zip(_np_gtlist, _np_probls):
            corr = np.sum(gt == gtprob)
            acc = corr*1.0/all_points
            #print(acc)
            acc_list.append(round(float(acc), precision))

        acc = round(float(np.mean(acc_list)), precision)
        #print('acc=',acc)
        if need_all:
            return acc, acc_list
        else:
            return acc
import pysnooper
#@pysnooper.snoop()
def get_score(_np_gtlist,_np_probls, need_all=False,precision=3):
        """
        正确率公式： AC = (TP+TN)/(TP+TN+FP+FN)
        # TP : True Positive , TN: True Negative, FP: False Positive, FN : False Negative,
        :return:预测结果和分割图一致的点的数目/全部的像素点数
        """
        all_points = np.prod(_np_gtlist[0].shape)
        #all_points=1000000
        sums_1=[]
        sums_2=[]
        #print(all_points)
        for gt, pred in zip(_np_gtlist, _np_probls):
            #print(acc)
            gt=np.array(gt,dtype=float)
            pred=np.array(pred,dtype=float)
            #print(gt)
            #print(pred)
            loss=np.array(pred - gt,dtype=float)
            #print(loss)
            sum_1=np.sum(np.abs(loss,dtype=float))
            #print(np.abs(pred - gt,dtype=float))
            sum_2=np.sum(np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (pred > 0)*1) * (1 - (gt > 0)*1)))
            #sum_2=np.sum(np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) )
            #print(np.max(gt))
            #print(np.max(pred))
            #print(sum_1)
            #print(sum_2)
            sums_1.append(float(sum_1))
            sums_2.append(float(sum_2))

        score = round(1-(float(sum(sums_1))/sum(sums_2)), precision)
        #print('acc=',acc)
        if need_all:
            return score
        else:
            return score
def get_precision(_np_gtlist,_np_probls, need_all=False,precision=3):
        """
        查准率公式： TP/(TP + FP)
        # TP : True Positive, FP : False Positive
        :param need_all: 为真的时候, 返回全部图像的对应测试结果
        :return:计算查准率
        """
        precision_list_0 = []
        precision_list_1 = []
        precision_list_2 = []
        precision_list_3 = []
        num0=0
        num1=0
        num2=0
        num3=0
        for gt, gtprob in zip(_np_gtlist, _np_probls):
            _TP_0 = ((gtprob == 0)*1 + (gt == 0)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP_0 = ((gtprob == 0)*1 + (gt !=0)*1) == 2
            _PC_0 = float(np.sum(_TP_0)) / (float(np.sum(_TP_0 + _FP_0)) + 1e-6)
            precision_list_0.append(round(float(_PC_0), precision))
            
            _TP_1 = ((gtprob == 1)*1 + (gt == 1)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP_1 = ((gtprob == 1)*1 + (gt !=1)*1) == 2
            _PC_1 = float(np.sum(_TP_1)) / (float(np.sum(_TP_1 + _FP_1)) + 1e-6)
            precision_list_1.append(round(float(_PC_1), precision))

            _TP_2 = ((gtprob == 2)*1 + (gt == 2)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP_2 = ((gtprob == 2)*1 + (gt !=2)*1) == 2
            _PC_2 = float(np.sum(_TP_2)) / (float(np.sum(_TP_2 + _FP_2)) + 1e-6)
            precision_list_2.append(round(float(_PC_2), precision))

            _TP_3 = ((gtprob == 3)*1 + (gt == 3)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP_3 = ((gtprob == 3)*1 + (gt !=3)*1) == 2
            _PC_3 = float(np.sum(_TP_3)) / (float(np.sum(_TP_3 + _FP_3)) + 1e-6)
            precision_list_3.append(round(float(_PC_3), precision))
            if 0 in gt:
                num0=num0+1
            if 1 in gt:
                num1=num1+1
            if 2 in gt:
                num2=num2+1
            if 3 in gt:
                num3=num3+1
        precision_0 = round(float(np.mean(precision_list_0)), precision)
        precision_1 = round(float(np.mean(precision_list_1)), precision)
        precision_2 = round(float(np.mean(precision_list_2)), precision)
        precision_3 = round(float(np.mean(precision_list_3)), precision)
        if need_all:
            return precision_0, precision_list_0
        else:
            return [precision_0,precision_1,precision_2,precision_3]

def get_sensitivity(_np_gtlist,_np_probls, need_all=False,precision=3):
        """
        查准率/召回率/TPR【真正例率】/敏感性 的计算公式： SE = TPR = TP/(TP + FN)
        # TP : True Positive , FN : False Negative
        :return:计算查全率/召回率==recall
        """
        recall_list = []
        for gt, gtprob in zip(_np_gtlist, _np_probls):
            _TP = ((gtprob == 1)*1 + (gt == 1)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FN = ((gtprob == 0)*1 + (gt == 1)*1) == 2
            _SE = 1.0*np.sum(_TP) / (float(np.sum(_TP + _FN)) + 1e-6)
            recall_list.append(round(float(_SE), precision))

        recall = round(float(np.mean(recall_list)), precision)
        if need_all:
            return recall, recall_list
        else:
            return recall

def get_specificity(_np_gtlist,_np_probls, need_all=False,precision=3):
        """
        特异性specificity/假正例率FPR  的计算公式： SP = FPR = FP/(TN + FP)
        # TN : True Negative, FP : False Positive
        :param need_all: 为真的时候, 返回全部图像的对应测试结果
        :return:计算特异性,或者说假正例率
        """
        specificity_list = []
        for gt, gtprob in zip(_np_gtlist, _np_probls):
            _TN = ((gtprob == 0)*1 + (gt == 0)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP = ((gtprob == 1)*1 + (gt == 0)*1) == 2
            _SP = float(np.sum(_TN)) / (float(np.sum(_TN + _FP)) + 1e-6)
            specificity_list.append(round(float(_SP), precision))

        specificity = round(float(np.mean(specificity_list)), precision)
        if need_all:
            return specificity, specificity_list
        else:
            return specificity

masks=[]
preds=[]
with torch.no_grad():
    for w,sample in enumerate(test_dataloader):
        input=sample['image']
        mask=sample['mask']
        img_2=sample['img_2']
        input_var = torch.autograd.Variable(input).float().cuda()
        input_var_2 = torch.autograd.Variable(img_2).float().cuda()
        #input_var = torch.autograd.Variable(input).float()
        result=model(input_var,input_var_2)
        #print(result.size())
        result = cv2.resize(result[0, :4].data.cpu().numpy().transpose(1, 2, 0),
                             (1000,1000),
                             interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
        mask=cv2.resize(mask.data.cpu().numpy().transpose(1, 2, 0),(1000,1000)).astype(np.uint8)
        masks.append(mask)
        preds.append(result)
        #print(get_score(_np_gtlist=mask,_np_probls=result))

        #print(np.shape(result))
        #print(np.shape(mask))
        #print(get_precision(_np_gtlist=mask,_np_probls=result))
        #print((result==mask).all())
        print(w)
accuracy=get_accuracy(_np_gtlist=masks,_np_probls=preds)      #正确率
#precision=get_precision(_np_gtlist=masks,_np_probls=preds)     #查全率
#sensitivity=get_sensitivity(_np_gtlist=masks,_np_probls=preds) #召回率 
#specificity=get_specificity(_np_gtlist=masks,_np_probls=preds) #特异性
score=get_score(_np_gtlist=masks,_np_probls=preds) 
print('acc=',accuracy)
print('score',score)



