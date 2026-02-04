import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re
  


class ResConv2D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv2D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re  
    
class Transient2volumn_grayd1(nn.Module):
    def __init__(self, nf0, in_channels, \
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn_grayd1, self).__init__()
        ###############################################
        # assert in_channels == 1
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        # weights[:, :, :, :, :] = 1.0
        
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )
    def forward(self, x0):
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=1, padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        re = torch.cat([x0_conv, x1], dim=1)
        return re
    

class Transient2volumn_gray(nn.Module):
    def __init__(self, nf0, in_channels, \
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn_gray, self).__init__()
        ###############################################
        # assert in_channels == 1
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[2, 2, 2],
                      bias=True),
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )
    def forward(self, x0):
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=2, padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        re = torch.cat([x0_conv, x1], dim=1)
        return re
    


class Transient2volumn_gray_nam(nn.Module):
    def __init__(self, nf0, in_channels, \
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn_gray_nam, self).__init__()
        ###############################################
        # assert in_channels == 1
        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        ##############################################
        self.conv1 = nn.Sequential(
            # begin, no norm
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[2, 2, 2],
                      bias=True),
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )
    def forward(self, x0):
        # x0 is from 0 to 1
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=2, padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        # re = torch.cat([x0_conv, x1], dim=1)
        re = x0_conv + x1
        return re

class Projector(nn.Module):
    """
    A projection module that projects 3D feature volume into 2D feature maps.
    """
    def __init__(self):
        super(Projector, self).__init__()

    def forward(self, x):
        # input_size  # (bs*v, c, d, hi, wi)
        x, idx = x.max(2)
        d = x.size(2)
        # larger value for closer planes
        depth = (d - 1 - idx.float()) / (d - 1)
        out = torch.cat([x, depth], dim=1)
        return out
    
class refinemodule(nn.Module):
    def __init__(self,refine_chann):
        super(refinemodule, self).__init__()
        self.refine = nn.Sequential(nn.Conv3d(refine_chann, refine_chann*2, 7, stride=(1,1,1), padding=3, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann*2,False),
                                    nn.Conv3d(refine_chann*2, refine_chann*2, 5, stride=(1,1,1), padding=2, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann*2,False),
                                    nn.Conv3d(refine_chann*2, refine_chann, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann,False),)
    def forward(self, x):
        out = self.refine(x)
        return out

class refinemodule_simple(nn.Module):
    def __init__(self,refine_chann):
        super(refinemodule_simple, self).__init__()
        self.refine = nn.Sequential(nn.Conv3d(refine_chann, refine_chann*2, kernel_size=3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann*2,False),
                                    nn.Conv3d(refine_chann*2, refine_chann, kernel_size=3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(refine_chann,False),
                                    )
    def forward(self, x):
        out = self.refine(x)
        return out
    
class Rendering(nn.Module):
    
    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()
        
        ######################################
        # assert out_channels == 1
        self.out_channels = out_channels
        weights = np.zeros((out_channels, out_channels*2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, out_channels:, :, :] = 1.0
        else:
            weights[:, :out_channels, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)
        
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + out_channels,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
    
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:self.out_channels, :, :]
        x0_dep = x0[:, dim:dim + self.out_channels, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + 1 * x2
        
        return re
    
class Interpsacle2d(nn.Module):
    
    def __init__(self, factor=2, gain=1, align_corners=False):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super(Interpsacle2d, self).__init__()
        self.gain = gain
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        
        x = nn.functional.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=self.align_corners)
        
        return x
    
    

   
class Rendering_up1(nn.Module):
    
    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering_up1, self).__init__()
        
        ######################################
        # assert out_channels == 1
        self.out_channels = out_channels
        weights = np.zeros((out_channels, out_channels*2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, out_channels:, :, :] = 1.0
        else:
            weights[:, :out_channels, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)
        
        self.resize = Interpsacle2d(factor=1, gain=1, align_corners=False)
        
        #######################################
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + out_channels,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )
    
    def forward(self, x0):
        
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:self.out_channels, :, :]
        x0_dep = x0[:, dim:dim + self.out_channels, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        
        ###################################
        x1 = self.conv1(x0)
        x1_up = x1#self.resize(x1)
        
        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)
        
        re = x0_conv_256 + 1 * x2
        
        return re
