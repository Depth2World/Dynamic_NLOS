
from math import sqrt, pow, log2
from operator import pos
import numpy as np
from numpy.core.shape_base import block
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.activation import Hardswish
from torch.nn.modules.container import T
from torch.nn.modules.normalization import CrossMapLRN2d
from .network_sup import *




class mask_conv_fusion(nn.Module):
    def __init__(self, in_chans=1,temp_chans=32,sp_ds_scale=8,targrt_size=128,inplace=False):
        super(mask_conv_fusion, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.targrt_size =targrt_size
        self.inplace = inplace
        self.sp_ds_scale = sp_ds_scale
        self.conv1 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,17,17), stride=(1,1,1), padding=(1,8,8), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv2 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,9,9), stride=(1,1,1), padding=(1,4,4), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv3 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,5,5), stride=(1,1,1), padding=(1,2,2), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv4 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,3,3), stride=(1,1,1), padding=(1,1,1), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        # self.conv2 = self.conv1
        # self.conv3 = self.conv1
        # self.conv4 = self.conv1
        
        self.fusion = nn.Sequential(nn.Conv3d(self.temp_chans*4, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.temp_chans*2,self.inplace),
                                    nn.Conv3d(self.temp_chans*2, self.temp_chans*1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.temp_chans*1,self.inplace),
                                    nn.Conv3d(self.temp_chans*1, self.in_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.in_chans,self.inplace),)
        
    def forward(self, mask_mea):
        mea_1 = self.conv1(mask_mea)
        mea_2 = self.conv2(mask_mea)
        mea_3 = self.conv3(mask_mea)
        mea_4 = self.conv4(mask_mea)
        
        # if self.sp_ds_scale >=1:
        #     b, c, t, h, w = mask_mea.shape
        #     ds_mea_reshape = mask_mea.reshape(-1,t,h,w)
        #     ds_mea_reshape = ds_mea_reshape[:,:,self.sp_ds_scale//2::self.sp_ds_scale,self.sp_ds_scale//2::self.sp_ds_scale]
        #     simple_up = F.interpolate(ds_mea_reshape, size=(self.targrt_size,) * 2) # , mode='nearest'
        #     simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
        #     # final+=simple_up
        
        mea_fusion = torch.cat([ mea_4,mea_3,mea_2,mea_1],dim=1)
        # mea_fusion = torch.cat([simple_up,mea_1],dim=1)
        final = self.fusion(mea_fusion)  
        return final
    


class mask_conv_fusion2(nn.Module):
    def __init__(self, in_chans=1,temp_chans=32,sp_ds_scale=8,targrt_size=128,inplace=False):
        super(mask_conv_fusion2, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.targrt_size =targrt_size
        self.inplace = inplace
        self.sp_ds_scale = sp_ds_scale
        self.conv1 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,17,17), stride=(2,1,1), padding=(1,8,8), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv2 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,9,9), stride=(2,1,1), padding=(1,4,4), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv3 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,5,5), stride=(2,1,1), padding=(1,2,2), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        self.conv4 = nn.Sequential(nn.Conv3d(self.in_chans, self.temp_chans, (3,3,3), stride=(2,1,1), padding=(1,1,1), dilation=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                   ResConv3D(self.temp_chans,self.inplace),)
        
        # self.conv2 = self.conv1
        # self.conv3 = self.conv1
        # self.conv4 = self.conv1
        
        self.fusion = nn.Sequential(nn.Conv3d(self.temp_chans*4, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.temp_chans*2,self.inplace),
                                    nn.Conv3d(self.temp_chans*2, self.temp_chans*1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.temp_chans*1,self.inplace),
                                    nn.Conv3d(self.temp_chans*1, self.in_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(self.in_chans,self.inplace),)
        
    def forward(self, mask_mea):
        mea_1 = self.conv1(mask_mea)
        mea_2 = self.conv2(mask_mea)
        mea_3 = self.conv3(mask_mea)
        mea_4 = self.conv4(mask_mea)
        
        # if self.sp_ds_scale >=1:
        #     b, c, t, h, w = mask_mea.shape
        #     ds_mea_reshape = mask_mea.reshape(-1,t,h,w)
        #     ds_mea_reshape = ds_mea_reshape[:,:,self.sp_ds_scale//2::self.sp_ds_scale,self.sp_ds_scale//2::self.sp_ds_scale]
        #     simple_up = F.interpolate(ds_mea_reshape, size=(self.targrt_size,) * 2) # , mode='nearest'
        #     simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
        #     # final+=simple_up
        
        mea_fusion = torch.cat([ mea_4,mea_3,mea_2,mea_1],dim=1)
        # mea_fusion = torch.cat([simple_up,mea_1],dim=1)
        final = self.fusion(mea_fusion)  
        return final


class TEMUP(nn.Module):
    r"""The TEMUP block
    It consists of several deconvolution block to upsample temporal dimension of input features while keep the spatial dimension.
    Input: ch_in, Ns (scales to be upsampled), (B, C, D, H, W)
    Output: (B, 1024, H, W)
    """
    def __init__(self, ch_in, Ns=16,ch_out=1):
        super().__init__()

        self.in_chans = ch_in
        self.Ns = int(log2(Ns))

        upblk = []
        for n in range(self.Ns - 1):
            in_chans_cur = self.in_chans // (pow(2,n))
            block_ft = nn.Sequential(
                nn.ConvTranspose3d(int(in_chans_cur), int(in_chans_cur//2), kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1),bias=False),
                nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(block_ft[0].weight, 0, 'fan_in', 'relu')
            upblk.append(block_ft)

        if self.in_chans // (pow(2,self.Ns-1)) <= 1:
            chans_last = 1
        else:
            chans_last = int(self.in_chans // (pow(2,self.Ns-1)))
        block_re = nn.Sequential(
            nn.ConvTranspose3d(chans_last, ch_out, kernel_size=(6,3,3), stride=(2,1,1), padding=(2,1,1),bias=False)
        )
        init.normal_(block_re[0].weight, mean=0, std=0.001)
        upblk.append(block_re)
        self.temup = nn.Sequential(*upblk)

    def forward(self, inputs):
        r"""
        Upsample the temporal dimension of the input features.
        Input: (B, C, D, H, W)
        Output: (B, 1024, H, W)
        """

        out = self.temup(inputs)    # B,C,D,H,W
        # out = torch.squeeze(out, 1) # B,1024,H,W

        return out



# class SPUP(nn.Module):
#     """The SPUP block
#     Upsample spatial dim and keep the temporal dimension
#     Input: B, C, D, H, W
#     Output: B, C, D, H*up, W*up (default)
#     """
#     def __init__(self, ch_in, up=2):
#         super().__init__()

#         self.in_chans = ch_in
#         self.up = int(log2(up))
#         upblk = []
#         for n in range(self.up):
#             sp_upblk = nn.Sequential(
#             nn.ConvTranspose3d(self.in_chans, self.in_chans, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
#             nn.ReLU(inplace=True)
#             )
#             init.kaiming_normal_(sp_upblk[0].weight, 0, 'fan_in', 'relu')
#             upblk.append(sp_upblk)
#         self.upop = nn.Sequential(*upblk)
#     def forward(self, inputs):
#         out = self.upop(inputs)
#         return out


class SPUP(nn.Module):
    """The SPUP block
    Upsample spatial dim and keep the temporal dimension
    Input: B, C, D, H, W
    Output: B, C, D, H*up, W*up (default)
    """
    def __init__(self, ch_in, up=8, ch_out=1):
        super().__init__()

        self.in_chans = ch_in
        self.up = int(log2(up))
        upblk = []
        for n in range(self.up-1):
            in_chans_cur = self.in_chans // (pow(2,n))
            sp_upblk = nn.Sequential(
            nn.ConvTranspose3d(int(in_chans_cur), int(in_chans_cur//2), (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(sp_upblk[0].weight, 0, 'fan_in', 'relu')
            upblk.append(sp_upblk)

        if self.in_chans // (pow(2,self.up-1)) <= 1:
            chans_last = 1
        else:
            chans_last = int(self.in_chans // (pow(2,self.up-1)))
        block_re = nn.Sequential(
            nn.ConvTranspose3d(chans_last, ch_out, kernel_size=(3,6,6), stride=(1,2,2), padding=(1,2,2),bias=False)
        )
        upblk.append(block_re)
        init.normal_(block_re[0].weight, mean=0, std=0.001)
        self.upop = nn.Sequential(*upblk)
    def forward(self, inputs):
        out = self.upop(inputs)
        return out


class MsFeat_2(nn.Module):
    """The feature extraction module 
    Extract features with downsampled temporal dim
    Input: B, C, D, H, W
    Output: B, 32, D/4, H, W
    """

    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        # self.ds = nn.Sequential(nn.Conv3d(self.in_chans, self.in_chans, 7, stride=(2,1,1), padding=3, dilation=1, groups=1, bias=True), nn.ReLU(inplace=True))
        # init.kaiming_normal_(self.ds[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[0].bias, 0.0)

        self.conv1 = nn.Sequential(nn.Conv3d(self.in_chans, self.out_chans, 3, stride=(2,1,1), padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv3d(self.in_chans, self.out_chans, 3, stride=(2,1,1), padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv3d(self.out_chans, self.out_chans, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv3d(self.out_chans, self.out_chans, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

        print("\033[0;33;40m=> WARN: MsFeat block in/out channel sizes {} / {}\033[0m".format(self.in_chans, 4* self.out_chans))

    def forward(self, inputs):
        # ds = self.ds(inputs)
        ds = inputs
        conv1 = self.conv1(ds)
        conv2 = self.conv2(ds)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)
        # return torch.cat((conv1,conv4), 1) + torch.cat((conv2, conv3), 1)


class DsFusion(nn.Module):
    """The DsFusion block
    Downsample temporal dim and fuse the them along channel
    Input: B, Cin, D, H, W
    Output: B, Cout, D/4, H, W
    """

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.in_chans = ch_in
        self.out_chans = ch_out
        self.ds = nn.Sequential(
            nn.Conv3d(self.in_chans, self.in_chans, kernel_size=5,stride=(2,1,1),padding=(2,2,2),bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.in_chans, self.in_chans, kernel_size=5,stride=(2,1,1),padding=(2,2,2),bias=True),
            nn.ReLU(inplace=True)#,
            #nn.Conv3d(self.in_chans, self.in_chans, kernel_size=5,stride=(2,1,1),padding=(2,2,2),bias=True),   #  nyu2
            #nn.ReLU(inplace=True)                                                                               #  nyu2
        )
        init.kaiming_normal_(self.ds[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[0].bias, 0.0)
        # init.kaiming_normal_(self.ds[2].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[2].bias, 0.0)
        #init.kaiming_normal_(self.ds[4].weight, 0, 'fan_in', 'relu'); init.constant_(self.ds[4].bias, 0.0)   #  nyu2

        self.fusion = nn.Sequential(
            nn.Conv3d(self.in_chans, self.out_chans, kernel_size=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True)
        )
        init.kaiming_normal_(self.fusion[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.fusion[0].bias, 0.0)

        print("\033[0;33;40m=> WARN: DsFusion block in/out channel sizes and DS scale {} / {} / {}\033[0m".format(self.in_chans, self.out_chans, 4))
    
    def forward(self, inputs):
        ds = self.ds(inputs)
        out = self.fusion(ds)

        return out

    def flops(self, N):
        flops = 0
        # self.ds(inputs)
        flops += 2* self.in_chans* self.in_chans* self.ds[0].kernel_size[0]* self.ds[0].kernel_size[1]* self.ds[0].kernel_size[2]* N
        # self.fusion(ds)
        flops += self.in_chans* self.out_chans * N

        return flops

