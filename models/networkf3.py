import torch
import torch.nn as nn
from .utils_pytorch import fk_1_10
from .utils_pytorch import lct_1_10
from .utils_pytorch import phasor_1_10
from .utils_pytorch import helper
from .network_sup import *
from torch.nn.parallel import DistributedDataParallel as DDP
from pro.Loss import criterion_KL, criterion_L2, criterion_TV
import torch.nn.init as init
from math import sqrt, pow, log2

from .modules import *

def add_poisson_noise(signal, desired_snr_db):
    # 计算信号功率和所需的噪声功率
    signal_power = torch.mean(torch.mul(signal, signal))
    desired_snr = 10 ** (desired_snr_db / 10)  # 将dB转换为比值
    noise_power = signal_power / desired_snr

    # 计算泊松噪声的参数lambda（即平均率λ）
    noise_lambda = noise_power / 2  # 泊松分布的方差等于其均值

    # 生成泊松噪声
    noise = torch.distributions.Poisson(noise_lambda)

    # 将噪声添加到信号
    noisy_signal = signal + noise.sample(signal.shape)

    return noisy_signal



class mask_conv_fusionf3(nn.Module):
    def __init__(self, in_chans=1,temp_chans=32,sp_ds_scale=8,targrt_size=128,inplace=False):
        super(mask_conv_fusionf3, self).__init__()
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
                                    nn.Conv3d(self.temp_chans*1, 1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
                                    ResConv3D(1,self.inplace),)
        
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

class transient_upf3(nn.Module):
    def __init__(self, in_chans=1,temp_chans=8,sp_ds_scale=4,targrt_size=128,inplace=False):
        super(transient_upf3, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.sp_ds_scale = sp_ds_scale
        self.targrt_size =targrt_size
        self.inplace = inplace
        
        self.denoise = MsFeat_2(in_channels= self.in_chans, out_channels=self.temp_chans)   
        self.spup = SPUP(self.temp_chans*4,self.sp_ds_scale, 1)
        # self.temup = TEMUP(self.temp_chans*4,2,self.in_chans)

        
        # self.ref2 = nn.Sequential(
        #     nn.Conv3d(self.temp_chans*2, self.temp_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
        #     ResConv3D(self.temp_chans,self.inplace),
        #     nn.Conv3d(self.temp_chans, self.temp_chans//2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
        #     ResConv3D(self.temp_chans//2,self.inplace),
        #     nn.Conv3d(self.temp_chans//2,1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        # )
        
    def forward(self, ds_meas):
        b,c,t,h,w = ds_meas.shape
        denoise = self.denoise(ds_meas)  # b, 32, t/2 ,h , w  
        spup = self.spup(denoise)      # b, 32, t/2 ,h*self.sp_ds_scale, w*self.sp_ds_scale
        # temup = self.temup(spup)       # b, 32, t   ,h*self.sp_ds_scale, w*self.sp_ds_scale
        simple_up = F.interpolate(ds_meas[:,1,...].reshape(-1,t,h,w), size=(self.targrt_size,) * 2)
        simple_up = simple_up.reshape(b,1,t,self.targrt_size,self.targrt_size)
        simple_up = simple_up[:,:,::2,:,:] + simple_up[:,:,1::2,:,:]
        # print(simple_up.shape,spup.shape)
        final = spup + simple_up
        return final
    


class dy_nlos(nn.Module):
    def __init__(self,in_ch=1,out_ch=2, ds=8, spatial=128,tlen=256,bin_len=0.02,wall_size=2,frames=3, local_rank=-1):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.ds = ds
        self.spatial = spatial
        self.tlen = tlen
        self.bin_len = bin_len
        self.frames = frames
        self.local_rank = local_rank
        basedim = 3
        refine_chann = basedim + self.in_channels

        self.sig_sr = transient_upf3(in_chans=self.in_channels*self.frames,temp_chans=8,sp_ds_scale=self.ds,targrt_size=self.spatial,inplace=False)
        self.sig_pad = mask_conv_fusionf3(in_chans=self.in_channels*self.frames,temp_chans=8,sp_ds_scale=self.ds,targrt_size=self.spatial,inplace=False)
        
        self.downnet = Transient2volumn_grayd1(nf0=basedim, in_channels=self.in_channels)
        # self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=wall_size, sampling_coeff=2.0, cycles=5,dnum=basedim + self.in_channels)
        self.tra2vlo =  lct_1_10.lct(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=wall_size,method='lct',dnum=basedim + self.in_channels)
       
        self.visnet = Projector()   # 8    12  

        self.rendernet = Rendering_up1(nf0=refine_chann * 2, out_channels=self.out_channels // 2)
        self.depnet = Rendering_up1(nf0=refine_chann* 2, out_channels=self.out_channels // 2, isdep=True)
        self.loss = None
        self.preparam(wall_size,spatial,self.bin_len/299792458)
        self.register_buffer("rsd_field_todev", self.rsd_field, persistent=False)
        
    
    def preparam(self,wall_size,sptial_grid,bin_resolution=32e-12,sampling_coeff=3,cycles=5):
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # self.virtual_wavelength = virtual_wavelength
        virtual_cos_wave_k, virtual_sin_wave_k = \
        helper.waveconvparam(bin_resolution, virtual_wavelength, cycles)
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        # use pytorch conv to replace matlab conv
        self.rsd_field = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
    
    def normalize(self, data_bxcxdxhxw):   #  min max scaling
        b, c, d, h, w = data_bxcxdxhxw.shape
        data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        
        data_min = data_bxcxk.min(2, keepdim=True)[0]
        data_zmean = data_bxcxk - data_min
        
        # most are 0
        data_max = data_zmean.max(2, keepdim=True)[0]
        data_norm = data_zmean / (data_max + 1e-15)

        return data_norm.view(b, c, d, h, w)
    
    def forward(self, ds_meas, pad_meas):
        # if self.training:
            # input = add_poisson_noise(input,30)
        # print(input.shape)
        ## input shape b f 1 t h w 

        # b, f, _, t, h, w = input.shape
        # inputs = input.reshape(-1,1,t,h,w) #b*f,c,t,h,w
        
        sig_sr = self.sig_sr(ds_meas[:,:,0,...])
        sig_pad = self.sig_pad(pad_meas[:,:,0,...]) 
        # print(sig_sr.shape,sig_pad.shape)
        sig_exp = sig_sr + sig_pad
        # sig_exp = sig_pad
        # sig_exp = sig_sr
        ## v3
        middle = self.normalize(sig_exp)      #  b*f,1,t,h,w
        fea = self.downnet(middle)      #  b*f,c,t,h,w
        vlo = self.tra2vlo(fea, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b*f,c,t,h,w
        
        vlo_ref = nn.ReLU()(vlo)
        vlo_ref = self.normalize(vlo_ref)    #  b,4,200,64,64
        raw = self.visnet(vlo_ref) 
        rendered_img = self.rendernet(raw)  # (bs*v, 1/3, hi, wi) 
        rendered_depth = self.depnet(raw)
        rendered_img = torch.cat([rendered_img,rendered_depth],dim=1)
        rendered_img = torch.clamp(rendered_img, 0, 1)
        rendered_img = rendered_img * 2 - 1
        
        return sig_exp, vlo_ref, rendered_img # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
            

    def train_one_iteration(self,optimizer):
        ## forward  
        sig_exp, vlo_ref, intdep = self.forward(self.ds_meas,self.pad_meas)
        intdep = (intdep + 1) / 2 
        rendered_img = intdep[:, 0, ...].unsqueeze(1)
        rendered_depth = intdep[:, 1, ...].unsqueeze(1)
        self.rendered_img = rendered_img
        self.rendered_depth = rendered_depth
        errs = self.compute_train_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt,sig_exp,self.meas_gt)
        if not isinstance(errs, list) and not isinstance(errs, tuple):
            errs = [errs]
        loss = sum(errs)
        
        ## back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## optimzer 
    
    def test_one_iteration(self):
        ## forward  
        sig_exp, vlo_ref, intdep = self.forward(self.ds_meas,self.pad_meas)
        intdep = (intdep + 1) / 2 
        rendered_img = intdep[:, 0, ...].unsqueeze(1)
        rendered_depth = intdep[:, 1, ...].unsqueeze(1)
        self.rendered_img = rendered_img
        self.rendered_depth = rendered_depth
        
        ## loss compute 
        # errs = self.compute_loss(self.rendered_img, self.int_gt, self.rendered_depth, self.dep_gt)
        errs = self.compute_train_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt,sig_exp,self.meas_gt)
        
        if not isinstance(errs, list) and not isinstance(errs, tuple):
            errs = [errs]
        loss = sum(errs)
        
        
    def set_input(self,sequence):
        # raw_meas = sequence['raw_meas'].to(self.local_rank)
        self.meas_gt = sequence['raw_meas'][:,1,...]
        self.meas_gt = self.meas_gt[:,:,::2,:,:] + self.meas_gt[:,:,1::2,:,:]
        self.ds_meas = sequence['ds_meas'] 
        self.pad_meas = sequence['pad_meas']
        self.int_gt = sequence['img_gt'][:,1,...]
        self.dep_gt = sequence['dep_gt'][:,1,...]
    
  
    def evluation_rw(self,ds_rw,pad_rw):
        self.sig_exp_re, self.vlo, intdep = self.forward(ds_rw,pad_rw)
        intdep = (intdep + 1) / 2 
        rendered_img = intdep[:, 0, ...].unsqueeze(1)
        rendered_depth = intdep[:, 1, ...].unsqueeze(1)
        self.rendered_img = rendered_img
        self.rendered_depth = rendered_depth
        
    def compute_train_loss(self,input0,gt0,input1,gt1,input2,gt2):
        self.vals = []
        rsd_loss = self.rsd_loss(input2,gt2) 
        meas_loss = criterion_L2(input2,gt2) 
        int_loss = criterion_L2(input0,gt0)
        dep_loss = criterion_L2(input1,gt1)
        tv_loss = criterion_TV(input0)

        self.vals.append(1*rsd_loss)
        self.vals.append(0.5*meas_loss)
        self.vals.append(1*int_loss)
        self.vals.append(1*dep_loss)
        self.vals.append(0.00*tv_loss)
        
        return self.vals
    
    
    def compute_loss(self,input0,gt0,input1,gt1):
        self.vals = []
        int_loss = criterion_L2(input0,gt0) 
        self.vals.append(int_loss)
        dep_loss = criterion_L2(input1,gt1) 
        self.vals.append(0*dep_loss)
        return self.vals
    
    def get_loss(self):
        return self.vals
    
    def rsd_loss(self,pred_mea,gt_mea):
        bnum, dnum, tnum, hnum, wnum = gt_mea.shape
        pred_data = pred_mea.view(bnum * dnum, tnum, hnum, wnum).permute(0, 2, 3, 1).reshape(-1, 1, tnum)
        gt_data = gt_mea.view(bnum * dnum, tnum, hnum, wnum).permute(0, 2, 3, 1).reshape(-1, 1, tnum)

        knum = self.rsd_field_todev.shape[2]
        phasor_data_mea_BDHWx2x1T = F.conv1d(pred_data, self.rsd_field_todev, padding=knum // 2)
        phasor_data_gt_BDHWx2x1T = F.conv1d(gt_data, self.rsd_field_todev, padding=knum // 2)

        rsd_loss = criterion_L2(phasor_data_mea_BDHWx2x1T,phasor_data_gt_BDHWx2x1T) 
        return rsd_loss
