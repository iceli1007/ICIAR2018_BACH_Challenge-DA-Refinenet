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

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from utils.helpers import maybe_download
from utils.layer_factory import conv1x1, conv3x3, CRPBlock,RCUBlock,RRBBlock,RRBBlock_1
from torchsummary import summary
model_50_path='/data4/lzq/data/ICIAR_2018/bishe_ICIAR/light-weight-refinenet/models/resnet50-19c8e357.pth'
model_101_path='/data4/lzq/data/ICIAR_2018/bishe_ICIAR/light-weight-refinenet/models/resnet101-5d3b4d8f.pth'
model_152_path='/data4/lzq/data/ICIAR_2018/bishe_ICIAR/light-weight-refinenet/models/resnet152-b121ed2d.pth'
data_info = {
    7 : 'Person',
    21: 'VOC',
    40: 'NYU',
    60: 'Context'
    }

models_urls = {
    '50_person'  : 'https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download',
    '101_person' : 'https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download',
    '152_person' : 'https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download',

    '50_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/2E1KrdF2Rfc5khB/download',
    '101_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download',
    '152_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download',

    '50_nyu'     : 'https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download',
    '101_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download',
    '152_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download',

    '101_context': 'https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download',
    '152_context': 'https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download',

    '50_imagenet' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetLW(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(ResNetLW, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out


class ResNetLW_CA(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(ResNetLW_CA, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.GAP_32=nn.AdaptiveAvgPool2d(1)
        #self.GAP_63=nn.AdaptiveAvgPool2d(63)
        #self.GAP_125=nn.AdaptiveAvgPool2d(125)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.conv32_1=conv1x1(512,512,bias=False)
        self.conv32_2=conv1x1(512,256,bias=False)
        self.sigmoid=nn.Sigmoid()

        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.conv63_1=conv1x1(512,512,bias=False)
        self.conv63_2=conv1x1(512,256,bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.conv125_1=conv1x1(512,512,bias=False)
        self.conv125_2=conv1x1(512,256,bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x_3=torch.cat((x3,x4),1)
        x_3=self.GAP_32(x_3)
        x_3=self.conv32_1(x_3)
        x_3=self.relu(x_3)
        x_3=self.conv32_2(x_3)
        x_3=self.sigmoid(x_3)
        #x3=torch.mul(x3,x_3)
        x3=x3*x_3
        x3=x3+x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x_2=torch.cat((x2,x3),1)
        x_2=self.GAP_32(x_2)
        x_2=self.conv63_1(x_2)
        x_2=self.relu(x_2)
        x_2=self.conv63_2(x_2)
        x_2=self.sigmoid(x_2)
        #x3=torch.mul(x2,x_2)
        x2=x2*x_2    
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x_1=torch.cat((x1,x2),1)
        x_1=self.GAP_32(x_1)
        x_1=self.conv125_1(x_1)
        x_1=self.relu(x_1)
        x_1=self.conv125_2(x_1)
        x_1=self.sigmoid(x_1)
        #x1=torch.mul(x1,x_1)
        x1=x1*x_1
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out


class RefineNet(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(RefineNet, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)

        out = self.clf_conv(x1)
        return out

class RefineNet_CA(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(RefineNet_CA, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.GAP_32=nn.AdaptiveAvgPool2d(1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.conv32_1=conv1x1(512,512,bias=False)
        self.conv32_2=conv1x1(512,256,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.conv63_1=conv1x1(512,512,bias=False)
        self.conv63_2=conv1x1(512,256,bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.conv125_1=conv1x1(512,512,bias=False)
        self.conv125_2=conv1x1(512,256,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g4_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl5_dimred=conv3x3(64,256,bias=False)
        self.adapt_stage5_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage5_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.conv256_1=conv1x1(512,512,bias=False)
        self.conv256_2=conv1x1(512,256,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.mflow_conv_g5_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g5_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g5_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #l0=x
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        
        x_3=torch.cat((x3,x4),1)
        x_3=self.GAP_32(x_3)
        x_3=self.conv32_1(x_3)
        x_3=self.relu(x_3)
        x_3=self.conv32_2(x_3)
        x_3=self.sigmoid(x_3)
        #x3=torch.mul(x3,x_3)
        x3=x3*x_3
        x3=x4+x3
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x_2=torch.cat((x2,x3),1)
        x_2=self.GAP_32(x_2)
        x_2=self.conv63_1(x_2)
        x_2=self.relu(x_2)
        x_2=self.conv63_2(x_2)
        x_2=self.sigmoid(x_2)
        #x3=torch.mul(x2,x_2)
        x2=x2*x_2    
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x_1=torch.cat((x1,x2),1)
        x_1=self.GAP_32(x_1)
        x_1=self.conv125_1(x_1)
        x_1=self.relu(x_1)
        x_1=self.conv125_2(x_1)
        x_1=self.sigmoid(x_1)
        #x1=torch.mul(x1,x_1)
        x1=x1*x_1
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        '''
        x1=self.mflow_conv_g4_b3_joint_varout_dimred(x1)
        x1=nn.Upsample(size=l0.size()[2:], mode='bilinear', align_corners=True)(x1)
        x0 = self.p_ims1d2_outl5_dimred(l0)
        x0 = self.adapt_stage5_b(x0)
        x0 = self.adapt_stage5_b2_joint_varout_dimred(x0)
        x_0=torch.cat((x0,x1),1)
        x_0=self.GAP_32(x_0)
        x_0=self.conv256_1(x_0)
        x_0=self.relu(x_0)
        x_0=self.conv256_2(x_0)
        x_0=self.sigmoid(x_0)
        #x1=torch.mul(x1,x_1)
        x0=x0*x_0
        x0 = x0 + x1
        x0 = F.relu(x0)
        x0 = self.mflow_conv_g5_pool(x0)
        x0 = self.mflow_conv_g5_b(x0)
        '''
        x1 = self.do(x1)
        
        out = self.clf_conv(x1)
        return out
class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(Resnet, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l4
class ResNet152LW_double(nn.Module):
#我们对大patch均采用resnet50作为特征提取器则为【3，4，6，3】
    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(ResNet152LW_double, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.basic_block_1 = Resnet(block, layers, num_classes=4)
        self.basic_block_2=Resnet(block, layers, num_classes=4)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p_ims1d2_outl1_dimred_1 = conv1x1(2048, 512, bias=False)
        self.p_ims1d2_outl1_dimred_2 = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl2_dimred_1 = conv1x1(1024, 256, bias=False)
        self.p_ims1d2_outl2_dimred_2 = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred_1 = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl3_dimred_2 = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred_1 = conv1x1(256, 256, bias=False)
        self.p_ims1d2_outl4_dimred_2= conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        #self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)
        



    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        #self.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #elf.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth')
        #print(state_dict)
        #model_state_dict={k[7:]: v for k, v in state_dict.items()} 
        #print(model_state_dict)   
        #self.basic_block_1.load_state_dict(model_state_dict,strict=False)
        #self.basic_block_2.load_state_dict(model_state_dict,strict=False)
        
        #params = list(self.basic_block_1.named_parameters())
        #params=list(params)
        #print(params[1])

        x = self.basic_block_1.conv1(x)
        x = self.basic_block_1.bn1(x)
        x = self.basic_block_1.relu(x)
        x = self.basic_block_1.maxpool(x)

        l1 = self.basic_block_1.layer1(x)
        l2 = self.basic_block_1.layer2(l1)
        l3 = self.basic_block_1.layer3(l2)
        l4 = self.basic_block_1.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        
        y = self.basic_block_2.conv1(y)
        y = self.basic_block_2.bn1(y)
        y= self.basic_block_2.relu(y)
        y = self.basic_block_2.maxpool(y)

        l5=  self.basic_block_2.layer1(y)
        l6 = self.basic_block_2.layer2(l5)
        l7 = self.basic_block_2.layer3(l6)
        l8 = self.basic_block_2.layer4(l7)
        l8=self.do(l8)
        l7=self.do(l7)
        
        x4 = self.p_ims1d2_outl1_dimred_1(l4)
        x4 = self.relu(x4)
        x8 = self.p_ims1d2_outl1_dimred_2(l8)
        x8 = self.relu(x8)
        x4=x4+x8
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)


        
        x3 = self.p_ims1d2_outl2_dimred_1(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x7 = self.p_ims1d2_outl2_dimred_2(l7)
        x7 = self.relu(x7)
        x3 = x3 + x4+ x7
        #x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        

        x2 = self.p_ims1d2_outl3_dimred_1(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x6 = self.p_ims1d2_outl3_dimred_2(l6)
        x6 = self.relu(x6)
        x2 = x2 + x3+x6
        #x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred_1(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x5 = self.p_ims1d2_outl4_dimred_2(l5)
        x5 = self.relu(x5)
        x1 = x1 + x2+x5
        #x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        #x1 = self.mflow_conv_g3_b3_joint_varout_dimred(x1)
        x1=self.do(x1)
        out = self.clf_conv(x1)
        return out



class Refinenet_double(nn.Module):
#我们对大patch均采用resnet50作为特征提取器则为【3，4，6，3】
    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        self.layers_1=[3,4,23,3]
        super(Refinenet_double, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.basic_block_1 = Resnet(block, layers, num_classes=4)
        self.basic_block_2=Resnet(block, layers=[3,4,6,3], num_classes=4)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p_ims1d2_outl1_dimred_1 = conv1x1(2048, 512, bias=False)
        self.p_ims1d2_outl1_dimred_2 = conv1x1(2048, 512, bias=False)
        self.adapt_stage1_b_1 = self._make_rcu(512, 512, 2, 2)
        self.adapt_stage1_b_2 = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl2_dimred_1 = conv1x1(1024, 256, bias=False)
        self.p_ims1d2_outl2_dimred_2 = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b_2 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred_1 = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl3_dimred_2 = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b_2 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred_1 = conv1x1(256, 256, bias=False)
        self.p_ims1d2_outl4_dimred_2= conv1x1(256, 256, bias=False)
        self.adapt_stage4_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b_2 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        #self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)





    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        #self.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #elf.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth')
        #print(state_dict)
        #model_state_dict={k[7:]: v for k, v in state_dict.items()} 
        #print(model_state_dict)   
        #self.basic_block_1.load_state_dict(model_state_dict,strict=False)
        #self.basic_block_2.load_state_dict(model_state_dict,strict=False)
        
        #params = list(self.basic_block_1.named_parameters())
        #params=list(params)
        #print(params[1])

        x = self.basic_block_1.conv1(x)
        x = self.basic_block_1.bn1(x)
        x = self.basic_block_1.relu(x)
        x = self.basic_block_1.maxpool(x)

        l1 = self.basic_block_1.layer1(x)
        l2 = self.basic_block_1.layer2(l1)
        l3 = self.basic_block_1.layer3(l2)
        l4 = self.basic_block_1.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        
        y = self.basic_block_2.conv1(y)
        y = self.basic_block_2.bn1(y)
        y= self.basic_block_2.relu(y)
        y = self.basic_block_2.maxpool(y)

        l5=  self.basic_block_2.layer1(y)
        l6 = self.basic_block_2.layer2(l5)
        l7 = self.basic_block_2.layer3(l6)
        l8 = self.basic_block_2.layer4(l7)
        l8=self.do(l8)
        l7=self.do(l7)
        
        x4 = self.p_ims1d2_outl1_dimred_1(l4)
        x4=self.adapt_stage1_b_1(x4)
        x4 = self.relu(x4)
        x8 = self.p_ims1d2_outl1_dimred_2(l8)
        x8=self.adapt_stage1_b_2(x8)
        x8 = self.relu(x8)
        x4=x4+x8
        x4 = self.mflow_conv_g1_pool(x4)
        x4=self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)


        
        x3 = self.p_ims1d2_outl2_dimred_1(l3)
        x3=self.adapt_stage2_b_1(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x7 = self.p_ims1d2_outl2_dimred_2(l7)
        x7=self.adapt_stage2_b_2(x7)
        x7 = self.relu(x7)
        x3 = x3 + x4+ x7
        #x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3=self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        

        x2 = self.p_ims1d2_outl3_dimred_1(l2)
        x2=self.adapt_stage3_b_1(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x6 = self.p_ims1d2_outl3_dimred_2(l6)
        x6=self.adapt_stage3_b_2(x6)
        x6 = self.relu(x6)
        x2 = x2 + x3+x6
        #x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2=self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred_1(l1)
        x1=self.adapt_stage4_b_1(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x5 = self.p_ims1d2_outl4_dimred_2(l5)
        x5=self.adapt_stage4_b_1(x5)
        x5 = self.relu(x5)
        x1 = x1 + x2+x5
        #x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        #x1 = self.mflow_conv_g3_b3_joint_varout_dimred(x1)
        x1=self.do(x1)
        out = self.clf_conv(x1)
        return out

class Refinenet_double_concat(nn.Module):
#我们对大patch均采用resnet50作为特征提取器则为【3，4，6，3】
    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(Refinenet_double_concat, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.basic_block_1 = Resnet(block, layers, num_classes=4)
        self.basic_block_2=Resnet(block, layers=[3,4,6,3], num_classes=4)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p_ims1d2_outl1_dimred_1 = conv1x1(2048, 512, bias=False)
        self.p_ims1d2_outl1_dimred_2 = conv1x1(2048, 512, bias=False)
        self.adapt_stage1_b_1 = self._make_rcu(512, 512, 2, 2)
        self.adapt_stage1_b_2 = self._make_rcu(512, 512, 2, 2)
        self.stage1_conv_1=conv1x1(1024,1024,bias=False)
        self.stage1_conv_2=conv1x1(1024,512,bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl2_dimred_1 = conv1x1(1024, 256, bias=False)
        self.p_ims1d2_outl2_dimred_2 = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b_2 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.stage2_conv_1=conv1x1(768,768,bias=False)
        self.stage2_conv_2=conv1x1(768,256,bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred_1 = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl3_dimred_2 = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b_2 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.stage3_conv_1=conv1x1(768,768,bias=False)
        self.stage3_conv_2=conv1x1(768,256,bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred_1 = conv1x1(256, 256, bias=False)
        self.p_ims1d2_outl4_dimred_2= conv1x1(256, 256, bias=False)
        self.adapt_stage4_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b_2 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.stage4_conv_1=conv1x1(768,768,bias=False)
        self.stage4_conv_2=conv1x1(768,256,bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        #self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)





    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        #self.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #elf.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth')
        #print(state_dict)
        #model_state_dict={k[7:]: v for k, v in state_dict.items()} 
        #print(model_state_dict)   
        #self.basic_block_1.load_state_dict(model_state_dict,strict=False)
        #self.basic_block_2.load_state_dict(model_state_dict,strict=False)
        
        #params = list(self.basic_block_1.named_parameters())
        #params=list(params)
        #print(params[1])

        x = self.basic_block_1.conv1(x)
        x = self.basic_block_1.bn1(x)
        x = self.basic_block_1.relu(x)
        x = self.basic_block_1.maxpool(x)

        l1 = self.basic_block_1.layer1(x)
        l2 = self.basic_block_1.layer2(l1)
        l3 = self.basic_block_1.layer3(l2)
        l4 = self.basic_block_1.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        
        y = self.basic_block_2.conv1(y)
        y = self.basic_block_2.bn1(y)
        y= self.basic_block_2.relu(y)
        y = self.basic_block_2.maxpool(y)

        l5=  self.basic_block_2.layer1(y)
        l6 = self.basic_block_2.layer2(l5)
        l7 = self.basic_block_2.layer3(l6)
        l8 = self.basic_block_2.layer4(l7)
        l8=self.do(l8)
        l7=self.do(l7)
        
        x4 = self.p_ims1d2_outl1_dimred_1(l4)
        x4=self.adapt_stage1_b_1(x4)
        x4 = self.relu(x4)
        x8 = self.p_ims1d2_outl1_dimred_2(l8)
        x8=self.adapt_stage1_b_2(x8)
        x8 = self.relu(x8)
        x4=torch.cat((x4,x8),1)
        #x4=self.stage1_conv_1(x4)
        #x4 = self.relu(x4)
        x4=self.stage1_conv_2(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4=self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)


        
        x3 = self.p_ims1d2_outl2_dimred_1(l3)
        x3=self.adapt_stage2_b_1(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x7 = self.p_ims1d2_outl2_dimred_2(l7)
        x7=self.adapt_stage2_b_2(x7)
        x7 = self.relu(x7)
        #x3 = x3 + x4+ x7
        x3=torch.cat((x3,x4,x7),1)
        #3=self.stage2_conv_1(x3)
        #x3 = self.relu(x3)
        x3=self.stage2_conv_2(x3)
        x3 = self.relu(x3)
        #x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3=self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        

        x2 = self.p_ims1d2_outl3_dimred_1(l2)
        x2=self.adapt_stage3_b_1(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x6 = self.p_ims1d2_outl3_dimred_2(l6)
        x6=self.adapt_stage3_b_2(x6)
        x6 = self.relu(x6)
        #x2 = x2 + x3+x6
        x2=torch.cat((x2,x3,x6),1)
        #x2=self.stage3_conv_1(x2)
        #x2 = self.relu(x2)
        x2=self.stage3_conv_2(x2)
        x2 = self.relu(x2)
        #x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2=self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred_1(l1)
        x1=self.adapt_stage4_b_1(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x5 = self.p_ims1d2_outl4_dimred_2(l5)
        x5=self.adapt_stage4_b_1(x5)
        x5 = self.relu(x5)
        #x1 = x1 + x2+x5
        x1=torch.cat((x1,x2,x5),1)
        #x1=self.stage4_conv_1(x1)
        #x1 = self.relu(x1)
        x1=self.stage4_conv_2(x1)
        x1 = self.relu(x1)
        #x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        #x1 = self.mflow_conv_g3_b3_joint_varout_dimred(x1)
        x1=self.do(x1)
        out = self.clf_conv(x1)
        return out
class Refinenet_double_attention(nn.Module):
#我们对大patch均采用resnet50作为特征提取器则为【3，4，6，3】
    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(Refinenet_double_attention, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.basic_block_1 = Resnet(block, layers, num_classes=4)
        self.basic_block_2=Resnet(block, layers=[3,4,6,3], num_classes=4)
        self.sigmoid=nn.Sigmoid()
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p_ims1d2_outl1_dimred_1 = conv1x1(2048, 512, bias=False)
        self.p_ims1d2_outl1_dimred_2 = conv1x1(2048, 512, bias=False)
        self.adapt_stage1_b_1 = self._make_rcu(512, 512, 2, 2)
        self.adapt_stage1_b_2 = self._make_rcu(512, 512, 2, 2)
        #self.conv16_1=conv1x1(1024,512,bias=False)
        #self.conv16_2=conv1x1(512,512,bias=False)
        self.stage1_conv_2=conv1x1(1024,512,bias=False)
        self.GAP_32=nn.AdaptiveAvgPool2d(1)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl2_dimred_1 = conv1x1(1024, 256, bias=False)
        self.p_ims1d2_outl2_dimred_2 = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b_2 = self._make_rcu(256, 256, 2, 2)
        #self.conv32_1=conv1x1(768,256,bias=False)
        #self.conv32_2=conv1x1(256,256,bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred_1 = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl3_dimred_2 = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b_2 = self._make_rcu(256, 256, 2, 2)
        #self.conv63_1=conv1x1(768,256,bias=False)
        #self.conv63_2=conv1x1(256,256,bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred_1 = conv1x1(256, 256, bias=False)
        self.p_ims1d2_outl4_dimred_2= conv1x1(256, 256, bias=False)
        self.adapt_stage4_b_1 = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b_2 = self._make_rcu(256, 256, 2, 2)
        #self.conv125_1=conv1x1(768,256,bias=False)
        #self.conv125_2=conv1x1(256,256,bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.stage4_conv_2=conv1x1(512,256,bias=False)
        self.stage3_conv_2=conv1x1(512,256,bias=False)
        self.stage2_conv_2=conv1x1(512,256,bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        #self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)





    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        #self.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #elf.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth')
        #print(state_dict)
        #model_state_dict={k[7:]: v for k, v in state_dict.items()} 
        #print(model_state_dict)   
        #self.basic_block_1.load_state_dict(model_state_dict,strict=False)
        #self.basic_block_2.load_state_dict(model_state_dict,strict=False)
        
        #params = list(self.basic_block_1.named_parameters())
        #params=list(params)
        #print(params[1])

        x = self.basic_block_1.conv1(x)
        x = self.basic_block_1.bn1(x)
        x = self.basic_block_1.relu(x)
        x = self.basic_block_1.maxpool(x)

        l1 = self.basic_block_1.layer1(x)
        l2 = self.basic_block_1.layer2(l1)
        l3 = self.basic_block_1.layer3(l2)
        l4 = self.basic_block_1.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        
        y = self.basic_block_2.conv1(y)
        y = self.basic_block_2.bn1(y)
        y= self.basic_block_2.relu(y)
        y = self.basic_block_2.maxpool(y)

        l5=  self.basic_block_2.layer1(y)
        l6 = self.basic_block_2.layer2(l5)
        l7 = self.basic_block_2.layer3(l6)
        l8 = self.basic_block_2.layer4(l7)
        l8=self.do(l8)
        l7=self.do(l7)
        
        x4 = self.p_ims1d2_outl1_dimred_1(l4)
        x4=self.adapt_stage1_b_1(x4)
        x4 = self.relu(x4)
        x8 = self.p_ims1d2_outl1_dimred_2(l8)
        x8=self.adapt_stage1_b_2(x8)
        x8 = self.relu(x8)
        
        x4=torch.cat((x4,x8),1)
        x4=self.stage1_conv_2(x4)
        x4 = self.relu(x4)
        '''
        x_4=self.GAP_32(x_4)
        x_4=self.conv16_1(x_4)
        x_4=self.relu(x_4)
        x_4=self.conv16_2(x_4)
        x_4=self.sigmoid(x_4)
        #x3=torch.mul(x3,x_3)
        x4_mul=x4*x_4
        x4=x4+x4_mul
        '''
        

        x4 = self.mflow_conv_g1_pool(x4)
        x4=self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)


        
        x3 = self.p_ims1d2_outl2_dimred_1(l3)
        x3=self.adapt_stage2_b_1(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x7 = self.p_ims1d2_outl2_dimred_2(l7)
        x7=self.adapt_stage2_b_2(x7)
        x7 = self.relu(x7)
        #x3 = x3 + x4+ x7
        #x3=torch.cat((x3,x4,x7),1)
        #x3 = F.relu(x3)
        #x3=self.stage2_conv_2(x3)
        
        x_3=torch.cat((x3,x4,x7),1)
        x_3=self.GAP_32(x_3)
        x_3=self.conv32_1(x_3)
        x_3=self.relu(x_3)
        x_3=self.conv32_2(x_3)
        x_3=self.sigmoid(x_3)
        #x3=torch.cat((x3,x7),1)
        #x3=self.stage2_conv_2(x3)
        #x3=torch.mul(x3,x_3)
        x3_mul=x3*x_3
        x3=x4+x3_mul
        
        x3 = self.mflow_conv_g2_pool(x3)
        x3=self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        

        x2 = self.p_ims1d2_outl3_dimred_1(l2)
        x2=self.adapt_stage3_b_1(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x6 = self.p_ims1d2_outl3_dimred_2(l6)
        x6=self.adapt_stage3_b_2(x6)
        x6 = self.relu(x6)
        #x2 = x2 + x3+x6
        #x2=torch.cat((x2,x3,x6),1)
        #x2 = F.relu(x2)
        #x2=self.stage3_conv_2(x2)
        
        x_2=torch.cat((x2,x3,x6),1)
        x_2=self.GAP_32(x_2)
        x_2=self.conv63_1(x_2)
        x_2=self.relu(x_2)
        x_2=self.conv63_2(x_2)
        x_2=self.sigmoid(x_2)
        #x3=torch.mul(x3,x_3)
        #x2=torch.cat((x2,x6),1)
        #x2=self.stage3_conv_2(x2)
        x2_mul=x2*x_2
        x2=x3+x2_mul
        
        x2 = self.mflow_conv_g3_pool(x2)
        x2=self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred_1(l1)
        x1=self.adapt_stage4_b_1(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        
        x5 = self.p_ims1d2_outl4_dimred_2(l5)
        x5=self.adapt_stage4_b_1(x5)
        x5 = self.relu(x5)
        #x1 = x1 + x2+x5
        #x1=torch.cat((x1,x2,x5),1)
        #x1 = F.relu(x1)
        
        x_1=torch.cat((x1,x2,x5),1)
        x_1=self.GAP_32(x_1)
        x_1=self.conv125_1(x_1)
        x_1=self.relu(x_1)
        x_1=self.conv125_2(x_1)
        x_1=self.sigmoid(x_1)
        #x3=torch.mul(x3,x_3)
        #x1=torch.cat((x1,x5),1)
        #x1=self.stage4_conv_2(x1)
        x1_mul=x1*x_1
        x1=x2+x1_mul
        
        x1=x1+x2
        x1 = F.relu(x1)
        
        #x1=self.stage4_conv_2(x1)
        x1 = self.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        #x1 = self.mflow_conv_g3_b3_joint_varout_dimred(x1)
        x1=self.do(x1)
        out = self.clf_conv(x1)
        return out
class ResNetLW_double(nn.Module):
#我们对大patch均采用resnet50作为特征提取器则为【3，4，6，3】
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        self.layers_1=[3,4,6,3]
        super(ResNetLW_double, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.basic_block_1 = Resnet(block, layers, num_classes=4)
        self.basic_block_2=Resnet(block, layers=[3,4,6,3], num_classes=4)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        x = self.basic_block_1.conv1(x)
        x = self.basic_block_1.bn1(x)
        x = self.basic_block_1.relu(x)
        x = self.basic_block_1.maxpool(x)

        l1 = self.basic_block_1.layer1(x)
        l2 = self.basic_block_1.layer2(l1)
        l3 = self.basic_block_1.layer3(l2)
        l4 = self.basic_block_1.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        
        y = self.basic_block_2.conv1(y)
        y = self.basic_block_2.bn1(y)
        y= self.basic_block_2.relu(y)
        y = self.basic_block_2.maxpool(y)

        l5=  self.basic_block_2.layer1(y)
        l6 = self.basic_block_2.layer2(l5)
        l7 = self.basic_block_2.layer3(l6)
        l8 = self.basic_block_2.layer4(l7)
        l8=self.do(l8)
        l7=self.do(l7)
        
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x8 = self.p_ims1d2_outl1_dimred(l8)
        x8 = self.relu(x8)
        x4=x4+x8
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)


        
        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x7 = self.p_ims1d2_outl2_dimred(l7)
        x7 = self.relu(x7)
        x3 = x3 + x4+ x7
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x6 = self.p_ims1d2_outl3_dimred(l6)
        x6 = self.relu(x6)
        x2 = x2 + x3+x6
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x5 = self.p_ims1d2_outl4_dimred(l5)
        x5 = self.relu(x5)
        x1 = x1 + x2+x5
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1=self.do(x1)
        out = self.clf_conv(x1)
        return out
class Refinenet_mylt_double(nn.Module):
#我们对大patch均采用resnet50作为特征提取器则为【3，4，6，3】
    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        self.layers_1=[3,4,23,3]
        super(Refinenet_mylt_double, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.basic_block_1 = Resnet(block, layers, num_classes=4)
        self.basic_block_2=Resnet(block, layers=[3,4,6,3], num_classes=4)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p_ims1d2_outl1_dimred_1 = conv1x1(2048, 512, bias=False)
        self.p_ims1d2_outl1_dimred_2 = conv1x1(2048, 512, bias=False)
        self.adapt_stage1_b_1 = self._make_rrb(512, 512)
        self.adapt_stage1_b_2 = self._make_rrb(512, 512)
        #self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rrb(512, 512)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        self.p_ims1d2_outl2_dimred_1 = conv1x1(1024, 256, bias=False)
        self.p_ims1d2_outl2_dimred_2 = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b_1 = self._make_rrb(256, 256)
        self.adapt_stage2_b_2 = self._make_rrb(256, 256)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        #self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rrb(256, 256)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred_1 = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl3_dimred_2 = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b_1 = self._make_rrb(256, 256)
        self.adapt_stage3_b_2 = self._make_rrb(256, 256)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        #self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rrb(256, 256)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred_1 = conv1x1(256, 256, bias=False)
        self.p_ims1d2_outl4_dimred_2= conv1x1(256, 256, bias=False)
        self.adapt_stage4_b_1 = self._make_rrb(256, 256)
        self.adapt_stage4_b_2 = self._make_rrb(256, 256)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        #self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rrb(256, 256)
        #self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)





    
    def _make_rrb(self, in_planes, out_planes):
        layers = [RRBBlock_1(in_planes, out_planes)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x,y):
        #self.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #elf.basic_block_1.load_state_dict(torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth'),strict=False)
        #state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model.pth')
        #print(state_dict)
        #model_state_dict={k[7:]: v for k, v in state_dict.items()} 
        #print(model_state_dict)   
        #self.basic_block_1.load_state_dict(model_state_dict,strict=False)
        #self.basic_block_2.load_state_dict(model_state_dict,strict=False)
        
        #params = list(self.basic_block_1.named_parameters())
        #params=list(params)
        #print(params[1])

        x = self.basic_block_1.conv1(x)
        x = self.basic_block_1.bn1(x)
        x = self.basic_block_1.relu(x)
        x = self.basic_block_1.maxpool(x)

        l1 = self.basic_block_1.layer1(x)
        l2 = self.basic_block_1.layer2(l1)
        l3 = self.basic_block_1.layer3(l2)
        l4 = self.basic_block_1.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)
        
        y = self.basic_block_2.conv1(y)
        y = self.basic_block_2.bn1(y)
        y= self.basic_block_2.relu(y)
        y = self.basic_block_2.maxpool(y)

        l5=  self.basic_block_2.layer1(y)
        l6 = self.basic_block_2.layer2(l5)
        l7 = self.basic_block_2.layer3(l6)
        l8 = self.basic_block_2.layer4(l7)
        l8=self.do(l8)
        l7=self.do(l7)
        
        x4 = self.p_ims1d2_outl1_dimred_1(l4)
        x4=self.adapt_stage1_b_1(x4)
        x4 = self.relu(x4)
        x8 = self.p_ims1d2_outl1_dimred_2(l8)
        x8=self.adapt_stage1_b_2(x8)
        x8 = self.relu(x8)
        x4=x4+x8
        #x4 = self.mflow_conv_g1_pool(x4)
        x4=self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)


        
        x3 = self.p_ims1d2_outl2_dimred_1(l3)
        x3=self.adapt_stage2_b_1(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x7 = self.p_ims1d2_outl2_dimred_2(l7)
        x7=self.adapt_stage2_b_2(x7)
        x7 = self.relu(x7)
        x3 = x3 + x4+ x7
        #x3 = F.relu(x3)
        #x3 = self.mflow_conv_g2_pool(x3)
        x3=self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)
        
        

        x2 = self.p_ims1d2_outl3_dimred_1(l2)
        x2=self.adapt_stage3_b_1(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x6 = self.p_ims1d2_outl3_dimred_2(l6)
        x6=self.adapt_stage3_b_2(x6)
        x6 = self.relu(x6)
        x2 = x2 + x3+x6
        #x2 = F.relu(x2)
        #x2 = self.mflow_conv_g3_pool(x2)
        x2=self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred_1(l1)
        x1=self.adapt_stage4_b_1(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x5 = self.p_ims1d2_outl4_dimred_2(l5)
        x5=self.adapt_stage4_b_1(x5)
        x5 = self.relu(x5)
        x1 = x1 + x2+x5
        #x1 = F.relu(x1)
        #x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        #x1 = self.mflow_conv_g3_b3_joint_varout_dimred(x1)
        x1=self.do(x1)
        out = self.clf_conv(x1)
        return out

def rf_lw50(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = Refinenet_double(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    '''
    state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model_50.pth')
    model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(model_state_dict,strict=False)
    #state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model_50.pth')
    #model_state_dict={k[7:]: v for k, v in state_dict.items()}
    #model.load_state_dict(model_state_dict,strict=False)
    #params = list(model.named_parameters())
    #params=list(params)
    #print(params[1])
    
    if imagenet:
        key = '50_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '50_' + dataset.lower()
            key = 'rf_lw' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    '''
    state_dict=torch.load(model_50_path)
    #print(state_dict)
    #model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model_state_dict_1={'basic_block_1.'+k: v for k, v in state_dict.items()}
    model_state_dict_2={'basic_block_2.'+k: v for k, v in state_dict.items()}
    model_state_dict={}
    model_state_dict.update(model_state_dict_1)
    model_state_dict.update(model_state_dict_2)
    model.load_state_dict(model_state_dict,strict=False)
    params = list(model.named_parameters())
    params=list(params)
    print(params[1])
    return model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
def rf_lw101(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = Refinenet_double(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    '''
    state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model_101.pth')
    model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(model_state_dict,strict=False)
    params = list(model.named_parameters())
    params=list(params)
    print(params[1])
    #model_state_dict_1={k.replace(k[:7],'basic_block_1.'): v for k, v in state_dict.items()}
    #model_state_dict_2={k.replace(k[:7],'basic_block_2.'): v for k, v in state_dict.items()}
    #print(model)
    #for param_tensor in model.state_dict():
        #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #summary(model,input_size=(3,500,500))
    
    if imagenet:
        key = '101_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '101_' + dataset.lower()
            key = 'rf_lw' + bname
            url = models_urls[bname]
    
            model.load_state_dict(maybe_download(key, url), strict=False)
    '''
    state_dict_1=torch.load(model_50_path)
    state_dict_2=torch.load(model_101_path)
    #print(state_dict)
    #model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model_state_dict_1={'basic_block_1.'+k: v for k, v in state_dict_2.items()}
    model_state_dict_2={'basic_block_2.'+k: v for k, v in state_dict_1.items()}
    model_state_dict={}
    model_state_dict.update(model_state_dict_1)
    model_state_dict.update(model_state_dict_2)
    model.load_state_dict(model_state_dict,strict=False)
    params = list(model.named_parameters())
    params=list(params)
    print(params[1])
    
    return model
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#model_summary = ResNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
#summary(model_summary,(3,500,500))
def rf_lw152(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = Refinenet_double(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    #for param_tensor in model.state_dict():
     #   print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #summary(model,input_size=(3,500,500))
    #print(model)
    '''
    state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model_152.pth')
    model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(model_state_dict,strict=False)
    
    state_dict=torch.load('/data4/lzq/data/ICIAR_2018/bishe_ICIAR/resnet/pytorch-best-practice/checkpoints/model_152.pth')
    #print(state_dict)
    #model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model_state_dict_1={k.replace(k[:7],'basic_block_1.'): v for k, v in state_dict.items()}
    model_state_dict_2={k.replace(k[:7],'basic_block_2.'): v for k, v in state_dict.items()}
    model_state_dict={}
    model_state_dict.update(model_state_dict_1)
    model_state_dict.update(model_state_dict_2)

    model.load_state_dict(model_state_dict,strict=False)
    params = list(model.named_parameters())
    params=list(params)
    print(params[1])
    
    if imagenet:
        key = '152_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '152_' + dataset.lower()
            key = 'rf_lw' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
            #odel=torch.nn.DataParallel(model, device_ids=[0,2])
    '''
    state_dict_1=torch.load(model_50_path)
    state_dict_2=torch.load(model_152_path)
    #print(state_dict)
    #model_state_dict={k[7:]: v for k, v in state_dict.items()}
    model_state_dict_1={'basic_block_1.'+k: v for k, v in state_dict_2.items()}
    model_state_dict_2={'basic_block_2.'+k: v for k, v in state_dict_1.items()}
    model_state_dict={}
    model_state_dict.update(model_state_dict_1)
    model_state_dict.update(model_state_dict_2)
    model.load_state_dict(model_state_dict,strict=False)
    params = list(model.named_parameters())
    params=list(params)
    print(params[1])
    
    
    
    return model

