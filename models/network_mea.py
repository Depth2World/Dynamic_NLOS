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
        
        self.fusion = nn.Sequential(nn.Conv3d(self.temp_chans*4 +1, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
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
        
        if self.sp_ds_scale >=1:
            b, c, t, h, w = mask_mea.shape
            ds_mea_reshape = mask_mea.reshape(-1,t,h,w)
            ds_mea_reshape = ds_mea_reshape[:,:,self.sp_ds_scale//2::self.sp_ds_scale,self.sp_ds_scale//2::self.sp_ds_scale]
            simple_up = F.interpolate(ds_mea_reshape, size=(self.targrt_size,) * 2) # , mode='nearest'
            simple_up = simple_up.reshape(b,c,t,self.targrt_size,self.targrt_size)
            # final+=simple_up
        
        mea_fusion = torch.cat([simple_up, mea_4,mea_3,mea_2,mea_1],dim=1)
        # mea_fusion = torch.cat([simple_up,mea_1],dim=1)
        final = self.fusion(mea_fusion)  
        return final
   

class transient_up(nn.Module):
    def __init__(self, in_chans=1,temp_chans=32,sp_ds_scale=4,targrt_size=64,inplace=False):
        super(transient_up, self).__init__()
        self.in_chans = in_chans
        self.temp_chans = temp_chans
        self.targrt_size =targrt_size
        self.inplace = inplace
        self.denoise = nn.Sequential(
            nn.Conv3d(self.in_chans, self.temp_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            nn.Conv3d(self.temp_chans, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans*2,self.inplace),
            ResConv3D(self.temp_chans*2,self.inplace),
            )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(self.temp_chans*2, self.temp_chans*2, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.ReLU(inplace=True)
            )
            
        self.ref1 = nn.Sequential(
            nn.Conv3d(self.temp_chans*2, self.temp_chans*4, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans*4,self.inplace),
            nn.Conv3d(self.temp_chans*4, self.temp_chans*2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans*2,self.inplace),
            )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(self.temp_chans*2, self.temp_chans*2, (3,6,6), stride=(1,2,2),padding=(1,2,2),bias=False),
            nn.ReLU(inplace=True)
            )
        
        self.ref2 = nn.Sequential(
            nn.Conv3d(self.temp_chans*2, self.temp_chans, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans,self.inplace),
            nn.Conv3d(self.temp_chans, self.temp_chans//2, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            ResConv3D(self.temp_chans//2,self.inplace),
            nn.Conv3d(self.temp_chans//2,1, 3, stride=(1,1,1), padding=1, dilation=1, bias=True),
        )
        
    def forward(self, input):

        ##network
        b,c,t,h,w = input.shape
        denoise = self.denoise(input)

        # v1_0
        # up1 = self.up1(denoise)
        # ref1 = self.ref1(up1)
        # up2 = self.up2(ref1)
        # final = self.ref2(up2)
        # rup2 = F.upsample(input, [t,h*4,w*4], mode='trilinear')
        # final = final + rup2
        
        # # v1_1
        up1 = self.up1(denoise)
        ref1 = self.ref1(up1)

        rup1 = F.upsample(input, [t,h*2,w*2], mode='trilinear')
        ref1 = ref1 + rup1

        up2 = self.up2(ref1)

        final = self.ref2(up2)
        rup2 = F.upsample(input, [t,h*4,w*4], mode='trilinear')
        final = final + rup2

        
        #v1_2
        # up1 = F.upsample(denoise, [t,h*2,w*2], mode='trilinear')
        # ref1 = self.ref1(up1)

        # up2 = F.upsample(ref1, [t,h*4,w*4], mode='trilinear')

        # final = self.ref2(up2)

        # rup2 = F.upsample(input, [t,h*4,w*4], mode='trilinear')
        # final = final + rup2

        ## interp

        # up1 = self.up1(denoise)
        # ref1 = self.ref1(up1)
        # up2 = self.up2(ref1)
        # final = self.ref2(up2)

        # bf, c, t, h, w = input.shape
        # ds_mea_reshape = input.reshape(-1,t,h,w)
        # simple_up = F.interpolate(ds_mea_reshape, size=(self.targrt_size,) * 2, mode='nearest') 
        # simple_up = simple_up.reshape(bf,c,t,self.targrt_size,self.targrt_size)
        # final+=simple_up

        
        return final
        
    # def forward(self, input):
    #     ##network
    #     denoise = self.denoise(input)
    #     up1 = self.up1(denoise)
    #     ref1 = self.ref1(up1)
    #     up2 = self.up2(ref1)
    #     final = self.ref2(up2)

    #     return final
   
class dy_nlos(nn.Module):
    def __init__(self,in_ch=1,out_ch=2, ds=4, spatial=64,tlen=512,bin_len=0.02,wall_size=2,frames=3, local_rank=-1):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.spatial = spatial
        self.tlen = tlen
        self.bin_len = bin_len
        self.frames = frames
        self.local_rank = local_rank
        basedim = 3
        self.sig_exp = transient_up(in_chans=1,temp_chans=8,sp_ds_scale=4,targrt_size=self.spatial,inplace=False)
        
        # self.downnet = Transient2volumn_grayd1(nf0=basedim, in_channels=self.in_channels)
        # self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=2.0, sampling_coeff=2.0, cycles=5,dnum=basedim + self.in_channels)
        self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=wall_size, sampling_coeff=3.0, cycles=5,dnum=1)
        # self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,dnum=basedim + self.in_channels)
        # self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,dnum=1)
        # self.tra2vlo =  lct_1_10.lct(spatial=self.spatial, crop=self.tlen, wall_size=2.0, bin_len=bin_len,method='lct',dnum=basedim + self.in_channels)
        # self.tra2vlo =  lct_1_10.lct(spatial=self.spatial, crop=self.tlen, wall_size=2.0, bin_len=bin_len,method='lct',dnum=1)
        refine_chann = basedim + self.in_channels
        # self.refine_0 = refinemodule(refine_chann)

        self.rendernet = LastRefine(self.in_channels, self.out_channels//2)
        self.depnet = LastRefine(self.in_channels, self.out_channels//2)
        # self.rendernet = LastRefine3(self.in_channels, self.out_channels//2)
        # self.depnet = LastRefine3(self.in_channels, self.out_channels//2)
        self.loss = None
        self.preparam(wall_size,spatial)
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
    
    def forward(self, input):
        # print(input.shape)
        ## input shape b f 1 t h w 
        b, f, _, t, h, w = input.shape
        # inputs = input.reshape(-1,1,t,h,w) #b*f,c,t,h,w
        sig_exp = self.sig_exp(input[:,1,...]) #b*f,c,t,h,w
        # print(input.shape,sig_exp.shape)
        # sig_exp_re = sig_exp.reshape(b,f,*sig_exp.shape[1:])  #b f 1 t H W
        
        # if self.training:
        #     return sig_exp # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
        # else:
        #     middle = self.normalize(sig_exp)      #  b,c,t,H,W
        #     vlo_ref = self.tra2vlo(middle, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b,c,t,h,w
        #     rendered_img, idx = vlo_ref.max(2)
        #     d = vlo_ref.size(2)
        #     # larger value for closer planes
        #     rendered_depth = (d - 1 - idx.float()) / (d - 1)
        #     return sig_exp, vlo_ref, rendered_img, rendered_depth # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
        
        middle = self.normalize(sig_exp)      #  b,c,t,H,W
        vlo_ref = self.tra2vlo(middle, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b,c,t,h,w
        vlo_ref = nn.ReLU()(vlo_ref)
        vlo_ref = self.normalize(vlo_ref)    #  b,4,200,64,64

        raw_img, idx = vlo_ref.max(2)
        d = vlo_ref.size(2)
        # larger value for closer planes
        raw_depth = (d - 1 - idx.float()) / (d - 1)

        # front_view = raw_img.detach().cpu().numpy()[0, 0]
        # front_view = (front_view / np.max(front_view))*255
        # # print(front_view.shape)
        # import cv2
        # cv2.imwrite('/data/lihb/anotherdata/nlos_video_eccv2024/pred_int.png', front_view)


        rendered_img = self.rendernet(raw_img)
        rendered_depth = self.depnet(raw_depth)

        rendered_img = torch.clamp(rendered_img, 0, 1)
        rendered_img = rendered_img * 2 - 1

        rendered_depth = torch.clamp(rendered_depth, 0, 1)
        rendered_depth = rendered_depth * 2 - 1

        return sig_exp, vlo_ref, rendered_img, rendered_depth # torch.Size([1, 1, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
        


    def train_one_iteration(self,optimizer):
        ## forward  
        # sig_exp = self.forward(self.tran_in)
        sig_exp, vlo_ref, rendered_img, rendered_depth = self.forward(self.tran_in)
        self.rendered_img = (rendered_img + 1) / 2
        self.rendered_depth = (rendered_depth + 1) / 2
        
        ## loss compute 
        # errs = self.compute_train_loss(sig_exp,self.meas_gt)
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
        # print(data['ds_meas'].shape)
        sig_exp, vlo_ref, pred_img, pred_depth = self.forward(self.tran_in)
        rendered_img = (pred_img + 1) / 2
        rendered_depth = (pred_depth + 1) / 2
        self.rendered_img = rendered_img
        self.rendered_depth = rendered_depth
        
        # print(vlo_ref.shape, rendered_img.shape, rendered_depth.shape)
        # print(int_gt.shape, dep_gt.shape)
        
        ## loss compute 
        # errs = self.compute_train_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt,sig_exp,self.meas_gt)
        errs = self.compute_train_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt,sig_exp,self.meas_gt)
        # errs = self.compute_train_loss(sig_exp,self.meas_gt)
        
        if not isinstance(errs, list) and not isinstance(errs, tuple):
            errs = [errs]
        loss = sum(errs)
        
        
    def set_input(self,sequence):
        # raw_meas = sequence['raw_meas'].to(self.local_rank)
        self.meas_gt = sequence['raw_meas'][:,1,...]
        self.tran_in = sequence['ds_meas']
        self.int_gt = sequence['dep_gt'][:,1,...]
        self.dep_gt = sequence['img_gt'][:,1,...]
    
  
    def evluation_rw(self,rw_input):
        self.sig_exp_re, self.vlo, rendered_img, rendered_depth = self.forward(rw_input)
        self.rendered_img = (rendered_img + 1) / 2
        self.rendered_depth = (rendered_depth + 1) / 2
    
    # def compute_train_loss(self,input2,gt2):
    def compute_train_loss(self,input0,gt0,input1,gt1,input2,gt2):
        self.vals = []
        rsd_loss = self.rsd_loss(input2,gt2) 
        meas_loss =criterion_L2(input2,gt2) 
        int_loss = criterion_L2(input0,gt0)
        dep_loss = criterion_L2(input1,gt1)
        tv_loss = criterion_TV(input0)

        self.vals.append(1*rsd_loss)
        self.vals.append(0.5*meas_loss)
        self.vals.append(1*int_loss)
        self.vals.append(0*dep_loss)
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
    
    