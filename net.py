import numpy as np
import torch
import math
import torch.nn as nn
from scipy.misc import imread, imsave, imresize
import torch.nn.functional as F
import fusion_strategy

import fusion_strategy

class RefleConvRelu(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(RefleConvRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride, dilation=dilation, groups=groups))
        self.ac = nn.ReLU();
        self.ac2 = nn.Tanh();
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x, last = False):
        # print(x.size())
        if (last):
            return self.ac2(self.conv(x))
        else:
            return self.ac(self.conv(x))
  
#Information Probe A
class ReconVISnet(nn.Module):
    def __init__(self):
        super(ReconVISnet, self).__init__()

        kernel_size = 3
        stride = 1

        base_channels = 16
        in_channels = 32;
        out_channels_def = 32;
        out_channels_def2 = 64;

        self.CVIS1 = RefleConvRelu(1,16);
        self.CVIS2 = RefleConvRelu(16,32);
        self.CVIS3 = RefleConvRelu(32,16);
        self.CVIS4 = RefleConvRelu(16,1);

    
    def forward(self, fusion):    
        OCVIS1 = self.CVIS1(fusion);
        OCVIS2 = self.CVIS2(OCVIS1);
        OCVIS3 = self.CVIS3(OCVIS2);
        recVIS = self.CVIS4(OCVIS3,last = True);
        recVIS = recVIS / 2 + 0.5;         
        return recVIS;

#Information Probe B
class ReconIRnet(nn.Module):
    def __init__(self):
        super(ReconIRnet, self).__init__()

        kernel_size = 3
        stride = 1

        base_channels = 16
        in_channels = 32;
        out_channels_def = 32;
        out_channels_def2 = 64;

        self.CIR1 = RefleConvRelu(1,16);
        self.CIR2 = RefleConvRelu(16,32);
        self.CIR3 = RefleConvRelu(32,16);
        self.CIR4 = RefleConvRelu(16,1);
        
    
    def forward(self, fusion):    
        OCIR1 = self.CIR1(fusion);
        OCIR2 = self.CIR2(OCIR1);
        OCIR3 = self.CIR3(OCIR2);
        recIR = self.CIR4(OCIR3,last=True);
        recIR = recIR/2+0.5;        
        return recIR;

#ASE module
class ReconFuseNet(nn.Module):
    def __init__(self):
        super(ReconFuseNet, self).__init__()

        kernel_size = 3
        stride = 1

        base_channels = 16
        in_channels = 32;
        out_channels_def = 32;
        out_channels_def2 = 64;

        self.FIR = RefleConvRelu(1,32);
        self.FVIS = RefleConvRelu(1,32);
        
        self.FF1 = RefleConvRelu(64,32);
        self.FF2 = RefleConvRelu(32,16);
        self.FF3 = RefleConvRelu(16,1);
        
    
    def forward(self, recIR, recVIS):
        #Encoder forward

        OFIR = self.FIR(recIR);
        OFVIS = self.FVIS(recVIS);
        
        concatedFeatures = torch.cat([OFIR,OFVIS],1);
        
        OFF1 = self.FF1(concatedFeatures);
        OFF2 = self.FF2(OFF1);
        out = self.FF3(OFF2,last=True);
        
        out = out/2+0.5;        
        return out;