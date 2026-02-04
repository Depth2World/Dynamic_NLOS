from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import  numpy as np
import sys
from .helper import definePsf, resamplingOperator, waveconvparam, waveconv
import ipdb
from math import sqrt, pow, log2
import torch.nn.init as init

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
        # out = torch.cat([x, depth], dim=1)
        return x, depth


class pf_dnn(nn.Module):
    
    def __init__(self, spatial=128, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5,dnum=1):
        super(pf_dnn, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        # ipdb.set_trace()
        self.parpareparam()
        self.register_buffer("virtual_cos_sin_wave_inv_2x1xk_todev", self.virtual_cos_sin_wave_inv_2x1xk, persistent=False)
        datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32)
        self.register_buffer("datapad_2Dx2Tx2Hx2W", datapad_2Dx2Tx2Hx2W, persistent=False)
        self.register_buffer("mtx_MxM_todev", self.mtx_MxM, persistent=False)
        self.register_buffer("mtxi_MxM_todev", self.mtxi_MxM, persistent=False)
        self.register_buffer("invpsf_real_todev", self.invpsf_real, persistent=False)
        self.register_buffer("invpsf_imag_todev", self.invpsf_imag, persistent=False)
        self.register_buffer("gridz_1xMx1x1_todev", self.gridz_1xMx1x1, persistent=False)
        
        self.pf_refine = nn.Sequential(nn.Conv3d(2, 4, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(4,False),
                                    nn.Conv3d(4, 8, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(8,False),
                                    nn.Conv3d(8, 2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(2,False),)
        self.pro = Projector()
        self.int_refine = nn.Sequential(ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        )
        self.dep_refine = nn.Sequential(ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        )

    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution  # loght travel path 
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ## scale radiometric component
        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        virtual_cos_wave_k, virtual_sin_wave_k = waveconvparam(bin_resolution, virtual_wavelength, cycles)
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0) # 2 33 
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope) # 1024 256 256 
        fpsf = np.fft.fftn(psf) # 1024 256 256
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf) # 1024 256 256
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0) # 1 1024 256 256
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0) # 1 1024 256 256
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
        
    def forward(self, feture_bxdxtxhxw):
        # print(feture_bxdxtxhxw.shape)
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape # 1 1 512 128 128n
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        # 3 run lct radiometric term 
        data_BDxTxHxW = feture_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum) # bc,t,h,w
        gridz_1xMx1x1 = self.gridz_1xMx1x1_todev # 1 t, 1 , 1
        data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 1)
        feture_bxdxtxhxw = data_BDxTxHxW.view(bnum, dnum, tnum, hnum, wnum)
        # print('4 radiometric')
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        # ipdb.set_trace()
        K = self.virtual_cos_sin_wave_inv_2x1xk_todev.shape[-1]
        wave_kernel = self.virtual_cos_sin_wave_inv_2x1xk_todev.view(2, 1, K, 1, 1)
        # print(wave_kernel.shape)
        phasor_cube = F.conv3d(feture_bxdxtxhxw, weight=wave_kernel, padding=((K - 1) // 2, 0, 0)) # 1 2 512 128 128
        phasor_cube = self.pf_refine(phasor_cube)
        # print(phasor_cube.shape)

        # phasor_f = torch.fft.rfft(phasor_cube.float(), dim=2)[:, :, :-1]
        # phasor_f = phasor_f.sum(dim=1)
        # phasor_f = torch.view_as_real(phasor_f) # 1 256 128 128 2
        # phasor_f = phasor_f.permute(0, 4, 1, 2, 3) # 1 2 256 128 128
        
        data_2BDxTxHxW = phasor_cube.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W # 2 1024 256 256 
        
        left = self.mtx_MxM_todev # 512 512
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1) # 2 512 128*128
        tmp = torch.matmul(left, right) # 2 512 128*128
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid) # 2 512 128 128
        
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        datafre_temp = torch.fft.fftn(datapad_2BDx2Tx2Hx2W, dim = (-3,-2,-1))   # 2 1024 256 256
        refre = datafre_temp * torch.complex(self.invpsf_real_todev,self.invpsf_imag_todev)
        re_temp = torch.fft.ifftn(refre,dim=(-3,-2,-1)) # 2 1024 256 256
        
        # ########################################################################
        # # Step 4: compute phasor field magnitude and inverse LCT
        volumn_2BDxTxHxW = re_temp[:, :temprol_grid, :sptial_grid, :sptial_grid]
        cos = volumn_2BDxTxHxW[:bnum * dnum,...]
        sin = volumn_2BDxTxHxW[bnum * dnum:,...]
        sum = cos*cos + sin*sin  ## different with **2
        # tmp = (torch.sqrt(sum.real**2 + sum.imag**2) + sum.real)/2  # numerical issue
        tmp = sum.real  # same as matlab
        
        tmp = F.relu(tmp, inplace=False)  # 1 512 128 128
        sqrt_sum_real = torch.sqrt(tmp)
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)  # 1 512 128 128
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum) 
        # return volumn_BxDxTxHxW
        inten, depth = self.pro(volumn_BxDxTxHxW)
        
        # inten = self.int_refine(inten)
        # depth = self.dep_refine(depth)
        return inten, depth


class SPUP(nn.Module):
    """The SPUP block
    Upsample spatial dim and keep the temporal dimension
    Input: B, C, D, H, W
    Output: B, C, D, H*up, W*up (default)
    """

    def __init__(self, ch_in, up=2):
        super().__init__()

        self.in_chans = ch_in
        self.up = int(log2(up))

        upblk = []
        for n in range(self.up):
            sp_upblk = nn.Sequential(
            nn.ConvTranspose3d(self.in_chans, self.in_chans, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.ReLU(inplace=True)
            )
            init.kaiming_normal_(sp_upblk[0].weight, 0, 'fan_in', 'relu')
            upblk.append(sp_upblk)
        
        self.upop = nn.Sequential(*upblk)

    def forward(self, inputs):

        out = self.upop(inputs)

        return out


class pf_up_dnn(nn.Module):
    
    def __init__(self, spatial=16,target_spatial=128,crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5,dnum=1):
        super(pf_up_dnn, self).__init__()
        
        self.spatial_grid = spatial
        self.target_spatial_grid = target_spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        # ipdb.set_trace()
        self.parpareparam()
        self.register_buffer("virtual_cos_sin_wave_inv_2x1xk_todev", self.virtual_cos_sin_wave_inv_2x1xk, persistent=False)
        datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.target_spatial_grid, 2 * self.target_spatial_grid), dtype=torch.float32)
        self.register_buffer("datapad_2Dx2Tx2Hx2W", datapad_2Dx2Tx2Hx2W, persistent=False)
        self.register_buffer("mtx_MxM_todev", self.mtx_MxM, persistent=False)
        self.register_buffer("mtxi_MxM_todev", self.mtxi_MxM, persistent=False)
        self.register_buffer("invpsf_real_todev", self.invpsf_real, persistent=False)
        self.register_buffer("invpsf_imag_todev", self.invpsf_imag, persistent=False)
        self.register_buffer("gridz_1xMx1x1_todev", self.gridz_1xMx1x1, persistent=False)
        
        self.pf_refine = nn.Sequential(nn.Conv3d(2, 4, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(4,False),
                                    nn.Conv3d(4, 8, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(8,False),
                                    nn.Conv3d(8, 2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
                                    ResConv3D(2,False),)
        self.pf_up = SPUP(2, up=target_spatial//spatial)
        self.pro = Projector()
        self.int_refine = nn.Sequential(ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        )
        self.dep_refine = nn.Sequential(ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        ResConv2D(1, inplace=False),
                                        )

    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution  # loght travel path 
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        target_spatial_grid = self.target_spatial_grid

        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ## scale radiometric component
        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6

        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # s_lamda_limit = wall_size / (target_spatial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        virtual_cos_wave_k, virtual_sin_wave_k = waveconvparam(bin_resolution, virtual_wavelength, cycles)
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0) # 2 33 
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(target_spatial_grid, temprol_grid, slope) # 1024 256 256 
        fpsf = np.fft.fftn(psf) # 1024 256 256
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf) # 1024 256 256
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0) # 1 1024 256 256
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0) # 1 1024 256 256
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
        
    def forward(self, feture_bxdxtxhxw, raw_transient=None):
        # print(feture_bxdxtxhxw.shape)
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape # 1 1 512 128 128
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        # 3 run lct radiometric term 
        data_BDxTxHxW = feture_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum) # bc,t,h,w

        gridz_1xMx1x1 = self.gridz_1xMx1x1_todev # 1 t, 1 , 1
        data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 4)

        feture_bxdxtxhxw = data_BDxTxHxW.view(bnum, dnum, tnum, hnum, wnum)

        # print('4 radiometric')
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        # ipdb.set_trace()
        K = self.virtual_cos_sin_wave_inv_2x1xk_todev.shape[-1]
        wave_kernel = self.virtual_cos_sin_wave_inv_2x1xk_todev.view(2, 1, K, 1, 1)
        # print(wave_kernel.shape)
        phasor_cube = F.conv3d(feture_bxdxtxhxw, weight=wave_kernel, padding=((K - 1) // 2, 0, 0)) # 1 2 512 128 128
        if raw_transient!=None:
            _,_,t,h,w = raw_transient.shape
            raw_transient = raw_transient.view(bnum * dnum, t, h, w)
            raw_transient = raw_transient * (gridz_1xMx1x1 ** 4)
            raw_transient = raw_transient.view(bnum, dnum, t, h, w)
            
            raw_cube = F.conv3d(raw_transient, weight=wave_kernel, padding=((K - 1) // 2, 0, 0)) # 1 2 512 128 128

        # phasor_cube = self.pf_refine(phasor_cube)
        # print(phasor_cube.shape)
        phasor_cube = self.pf_up(phasor_cube)
        # print(phasor_cube.shape,raw_cube.shape)

        # phasor_f = torch.fft.rfft(phasor_cube.float(), dim=2)[:, :, :-1]
        # phasor_f = phasor_f.sum(dim=1)
        # phasor_f = torch.view_as_real(phasor_f) # 1 256 128 128 2
        # phasor_f = phasor_f.permute(0, 4, 1, 2, 3) # 1 2 256 128 128
        
        # data_2BDxTxHxW = phasor_cube.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        # data_2BDxTxHxW = phasor_cube.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        data_2BDxTxHxW = phasor_cube.reshape(2 * bnum * dnum, tnum, self.target_spatial_grid, self.target_spatial_grid)
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W # 2 1024 256 256 
        
        left = self.mtx_MxM_todev # 512 512
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1) # 2 512 128*128
        tmp = torch.matmul(left, right) # 2 512 128*128
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, self.target_spatial_grid, self.target_spatial_grid) # 2 512 128 128
        
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :self.target_spatial_grid, :self.target_spatial_grid] = tmp2
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        datafre_temp = torch.fft.fftn(datapad_2BDx2Tx2Hx2W, dim = (-3,-2,-1))   # 2 1024 256 256
        refre = datafre_temp * torch.complex(self.invpsf_real_todev,self.invpsf_imag_todev)
        re_temp = torch.fft.ifftn(refre,dim=(-3,-2,-1)) # 2 1024 256 256
        
        # ########################################################################
        # # Step 4: compute phasor field magnitude and inverse LCT
        volumn_2BDxTxHxW = re_temp[:, :temprol_grid, :self.target_spatial_grid, :self.target_spatial_grid]
        cos = volumn_2BDxTxHxW[:bnum * dnum,...]
        sin = volumn_2BDxTxHxW[bnum * dnum:,...]
        sum = cos*cos + sin*sin  ## different with **2
        # tmp = (torch.sqrt(sum.real**2 + sum.imag**2) + sum.real)/2  # numerical issue
        tmp = sum.real  # same as matlab
        
        tmp = F.relu(tmp, inplace=False)  # 1 512 128 128
        sqrt_sum_real = torch.sqrt(tmp)
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, self.target_spatial_grid, self.target_spatial_grid)  # 1 512 128 128
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        # volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum) 
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, self.target_spatial_grid, self.target_spatial_grid) 
        # return volumn_BxDxTxHxW
        inten, depth = self.pro(volumn_BxDxTxHxW)
        
        inten = self.int_refine(inten)
        depth = self.dep_refine(depth)
        if raw_transient!= None:
            return phasor_cube, raw_cube, inten, depth
        else:
            return inten, depth




if __name__ == '__main__':
    pass