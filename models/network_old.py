import torch
import torch.nn as nn
from .utils_pytorch import fk_1_10
from .utils_pytorch import lct_1_10
from .utils_pytorch import phasor_1_10
from .network_sup import *
from torch.nn.parallel import DistributedDataParallel as DDP
from pro.Loss import criterion_KL, criterion_L2
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
    
# class dy_nlos(nn.Module):
#     def __init__(self,in_ch=1,out_ch=2,spatial=64,tlen=512,bin_len=0.02,wall_size=2,frames=3, local_rank=-1):
#         super().__init__()
#         self.in_channels = in_ch
#         self.out_channels = out_ch
#         self.spatial = spatial
#         self.tlen = tlen
#         self.bin_len = bin_len
#         self.frames = frames
#         self.local_rank = local_rank
#         basedim = 3
#         self.cs_blk = mask_conv_fusion(in_chans=1,temp_chans=8,sp_ds_scale=4,targrt_size=64,inplace=False)
        
#         self.downnet = Transient2volumn_grayd1(nf0=basedim, in_channels=self.in_channels)
#         # self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=bin_len, wall_size=2.0, sampling_coeff=2.0, cycles=5,dnum=basedim + self.in_channels)
#         self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,dnum=basedim + self.in_channels)
#         # self.tra2vlo =  lct_1_10.lct(spatial=self.spatial, crop=self.tlen, wall_size=2.0, bin_len=bin_len,method='lct',dnum=basedim + self.in_channels)
#         refine_chann = basedim + self.in_channels
#         self.refine_0 = refinemodule(refine_chann)
#         self.refine_1 = refinemodule(refine_chann)
#         self.refine_2 = refinemodule(refine_chann)
#         self.visnet = Projector()   # 8    12  
#         self.rendernet = Rendering_up1(nf0=refine_chann * 2, out_channels=self.out_channels // 2)
#         self.depnet = Rendering_up1(nf0=refine_chann* 2, out_channels=self.out_channels // 2, isdep=True)
#         self.loss = None
        
    
#     def normalize(self, data_bxcxdxhxw):   #  min max scaling
#         b, c, d, h, w = data_bxcxdxhxw.shape
#         data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        
#         data_min = data_bxcxk.min(2, keepdim=True)[0]
#         data_zmean = data_bxcxk - data_min
        
#         # most are 0
#         data_max = data_zmean.max(2, keepdim=True)[0]
#         data_norm = data_zmean / (data_max + 1e-15)

#         return data_norm.view(b, c, d, h, w)
    
#     def forward(self, input):
#         # print(input.shape)
#         ## input shape b f 1 t h w 
#         b, f, _, t, h, w = input.shape
#         inputs = input.reshape(-1,1,t,h,w) #b*f,c,t,h,w
#         sig_exp = self.cs_blk(inputs) #b*f,c,t,h,w
#         # sig_exp_re = sig_exp.reshape(*input.shape)
        
#         middle = self.normalize(sig_exp)      #  b*f,1,t,h,w
#         fea = self.downnet(middle)      #  b*f,c,t,h,w
#         vlo = self.tra2vlo(fea, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b*f,c,t,h,w
#         vlo = vlo.reshape(b, f, *vlo.shape[1:]) #  b,f,c,t,h,w
#         # print(vlo[:,0,...].shape)
        
#         # vlo_ref0 = self.refine_0(vlo[:,0,...])
#         vlo_ref1 = self.refine_0(vlo[:,1,...])
#         # vlo_ref2 = self.refine_0(vlo[:,2,...])
#         # vlo_ref = vlo_ref0 + vlo_ref1 + vlo_ref2  #  b,c,t,h,w
#         vlo_ref = vlo_ref1  #  b,c,t,h,w
        
#         # zdim = vlo.shape[2]
#         # zdimnew = zdim * 100 // 128
#         # vlo = vlo[:, :, :zdimnew]
#         vlo_ref = nn.ReLU()(vlo_ref)
#         vlo_ref = self.normalize(vlo_ref)    #  b,4,200,64,64
        
#         raw = self.visnet(vlo_ref) 
#         rendered_img = self.rendernet(raw)  # (bs*v, 1/3, hi, wi) 
#         # rendered_img = rendered_img.reshape(inputs.shape[0], -1, *rendered_img.shape[-3:])         # (bs, v, 1/3, ho, wo)
#         rendered_depth = self.depnet(raw)
#         # rendered_depth = rendered_depth.reshape(inputs.shape[0], -1, *rendered_img.shape[-3:])         # (bs, v, 1/3, ho, wo)
#         # rendered_depth = torch.mean(rendered_depth,dim=2,keepdim=True)
#         if self.training:
#             # return vlo_ref, rendered_img, rendered_depth # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
#             return sig_exp, vlo_ref, rendered_img, rendered_depth # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
#         else:
#             return vlo_ref, rendered_img, rendered_depth # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
            

#     def train_one_iteration(self,optimizer):
#         ## forward  
#         sig_exp, vlo_ref, rendered_img, rendered_depth = self.forward(self.tran_in)
#         # vlo_ref, rendered_img, rendered_depth = self.forward(self.tran_in)
#         # print(vlo_ref.shape, rendered_img.shape, rendered_depth.shape)
#         # print(int_gt.shape, dep_gt.shape)
        
#         ## loss compute 
#         # print(sig_exp.shape)
#         # print(self.meas_gt.shape)
#         # print(self.meas_gt.reshape(-1,1,*self.meas_gt.shape[3:]).shape)
#         errs = self.compute_train_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt,\
#             self.normalize(sig_exp), self.normalize(self.meas_gt.reshape(-1,1,*self.meas_gt.shape[3:])))
#         # errs = self.compute_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt)
#         if not isinstance(errs, list) and not isinstance(errs, tuple):
#             errs = [errs]
#         loss = sum(errs)
        
#         ## back propagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         ## optimzer 
    
#     def test_one_iteration(self):
#         ## forward  
#         # print(data['ds_meas'].shape)
#         _, self.rendered_img, self.rendered_depth = self.forward(self.tran_in)
#         # print(vlo_ref.shape, rendered_img.shape, rendered_depth.shape)
#         # print(int_gt.shape, dep_gt.shape)
        
#         ## loss compute 
#         errs = self.compute_loss(self.rendered_img, self.int_gt, self.rendered_depth, self.dep_gt)
#         if not isinstance(errs, list) and not isinstance(errs, tuple):
#             errs = [errs]
#         loss = sum(errs)
        
        
#     def set_input(self,sequence):
#         # raw_meas = sequence['raw_meas'].to(self.local_rank)
#         self.meas_gt = sequence['raw_meas']
#         self.tran_in = sequence['ds_meas']
#         self.int_gt = sequence['dep_gt'][:,1,...]
#         self.dep_gt = sequence['img_gt'][:,1,...]
    
#     # def load_ddp(self):
#     #     for name in self.ddp_model_names:
#     #         if isinstance(name, str):
#     #             setattr(self, name, torch.nn.SyncBatchNorm.convert_sync_batchnorm(getattr(self, name)).to(self.local_rank))
#     #             setattr(self, name, DDP(getattr(self, name), device_ids = [self.local_rank], output_device = self.local_rank, broadcast_buffers = False))
        
#     def evluation_rw(self,rw_input):
#         self.vlo, self.rendered_img, self.rendered_depth = self.forward(rw_input)
    
    
#     def compute_train_loss(self,input0,gt0,input1,gt1,input2,gt2):
#         self.vals = []
#         int_loss = criterion_L2(input0,gt0) 
#         dep_loss = criterion_L2(input1,gt1)
#         meas_loss = criterion_L2(input2,gt2) 
        
#         self.vals.append(int_loss)
#         self.vals.append(0*dep_loss)
#         self.vals.append(meas_loss)
        
#         return self.vals
    
    
#     def compute_loss(self,input0,gt0,input1,gt1):
#         self.vals = []
#         int_loss = criterion_L2(input0,gt0) 
#         self.vals.append(int_loss)
#         dep_loss = criterion_L2(input1,gt1) 
#         self.vals.append(0*dep_loss)
#         return self.vals
    
#     def get_loss(self):
#         return self.vals
    



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
        denoise = self.denoise(input)
        up1 = self.up1(denoise)
        ref1 = self.ref1(up1)
        up2 = self.up2(ref1)
        final = self.ref2(up2)
        return final
    
   
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
        self.sig_exp = transient_up(in_chans=1,temp_chans=16,sp_ds_scale=4,targrt_size=64,inplace=False)
        
        self.downnet = Transient2volumn_grayd1(nf0=basedim, in_channels=self.in_channels)
        # self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=2.0, sampling_coeff=2.0, cycles=5,dnum=basedim + self.in_channels)
        self.tra2vlo = phasor_1_10.phasor(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len, wall_size=2.0, sampling_coeff=2.0, cycles=5,dnum=1)
        # self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,dnum=basedim + self.in_channels)
        # self.tra2vlo = fk_1_10.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,dnum=1)
        # self.tra2vlo =  lct_1_10.lct(spatial=self.spatial, crop=self.tlen, wall_size=2.0, bin_len=bin_len,method='lct',dnum=basedim + self.in_channels)
        refine_chann = basedim + self.in_channels
        self.refine_0 = refinemodule(refine_chann)
        self.refine_1 = refinemodule(refine_chann)
        self.refine_2 = refinemodule(refine_chann)
        self.visnet = Projector()   # 8    12  
        self.rendernet = Rendering_up1(nf0=refine_chann * 2, out_channels=self.out_channels // 2)
        self.depnet = Rendering_up1(nf0=refine_chann* 2, out_channels=self.out_channels // 2, isdep=True)
        self.loss = None
        
    
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
        inputs = input.reshape(-1,1,t,h,w) #b*f,c,t,h,w
        sig_exp = self.sig_exp(inputs) #b*f,c,t,h,w
        # print(input.shape,sig_exp.shape)
        sig_exp_re = sig_exp.reshape(b,f,*sig_exp.shape[1:])  #b f 1 t H W 
        middle = self.normalize(sig_exp_re[:,1,...])      #  b,c,t,H,W
        # fea = self.downnet(middle)      #  b*f,c,t,h,w
        # vlo = self.tra2vlo(fea, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b*f,c,t,h,w
        # # vlo = vlo.reshape(b, f, *vlo.shape[1:]) #  b,f,c,t,h,w
        # vlo_ref1 = self.refine_0(vlo)
        # vlo_ref = vlo_ref1  #  b,c,t,h,w
        
        # # zdim = vlo.shape[2]
        # # zdimnew = zdim * 100 // 128
        # # vlo = vlo[:, :, :zdimnew]
        # vlo_ref = nn.ReLU()(vlo_ref)
        # vlo_ref = self.normalize(vlo_ref)    #  b,4,200,64,64
        # raw = self.visnet(vlo_ref) 
        # rendered_img = self.rendernet(raw)  # (bs*v, 1/3, hi, wi) 
        # rendered_depth = self.depnet(raw)
        # # rendered_depth = torch.mean(rendered_depth,dim=2,keepdim=True)
        
        vlo_ref = self.tra2vlo(middle, [0,0,0], [self.tlen,self.tlen,self.tlen]) #  b,c,t,h,w
        # vlo = vlo.reshape(b, f, *vlo.shape[1:]) #  b,f,c,t,h,w

        rendered_img, idx = vlo_ref.max(2)
        d = vlo_ref.size(2)
        # larger value for closer planes
        rendered_depth = (d - 1 - idx.float()) / (d - 1)
        # rendered_depth = torch.mean(rendered_depth,dim=2,keepdim=True)
        if self.training:
            return sig_exp_re, vlo_ref, rendered_img, rendered_depth # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
        else:
            return sig_exp_re, vlo_ref, rendered_img, rendered_depth # torch.Size([1, 4, 512, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 1, 64, 64])
            

    def train_one_iteration(self,optimizer):
        ## forward  
        sig_exp, vlo_ref, rendered_img, rendered_depth = self.forward(self.tran_in)
        # vlo_ref, rendered_img, rendered_depth = self.forward(self.tran_in)
        # print(vlo_ref.shape, rendered_img.shape, rendered_depth.shape)
        # print(int_gt.shape, dep_gt.shape)
        
        ## loss compute 
        # print(sig_exp.shape)
        # print(self.meas_gt.shape)
        # print(self.meas_gt.reshape(-1,1,*self.meas_gt.shape[3:]).shape)
        errs = self.compute_train_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt,sig_exp,self.meas_gt)
            # self.normalize(sig_exp), self.normalize(self.meas_gt.reshape(-1,1,*self.meas_gt.shape[3:])))
        # errs = self.compute_loss(rendered_img, self.int_gt, rendered_depth, self.dep_gt)
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
        sig_exp, vlo_ref, self.rendered_img, self.rendered_depth = self.forward(self.tran_in)
        # print(vlo_ref.shape, rendered_img.shape, rendered_depth.shape)
        # print(int_gt.shape, dep_gt.shape)
        
        ## loss compute 
        # errs = self.compute_loss(self.rendered_img, self.int_gt, self.rendered_depth, self.dep_gt)
        errs = self.compute_train_loss(self.rendered_img, self.int_gt, self.rendered_depth, self.dep_gt,sig_exp,self.meas_gt)
        
        if not isinstance(errs, list) and not isinstance(errs, tuple):
            errs = [errs]
        loss = sum(errs)
        
        
    def set_input(self,sequence):
        # raw_meas = sequence['raw_meas'].to(self.local_rank)
        self.meas_gt = sequence['raw_meas']
        self.tran_in = sequence['ds_meas']
        self.int_gt = sequence['dep_gt'][:,1,...]
        self.dep_gt = sequence['img_gt'][:,1,...]
    
  
    def evluation_rw(self,rw_input):
        self.sig_exp_re, self.vlo, self.rendered_img, self.rendered_depth = self.forward(rw_input)
    
    def compute_train_loss(self,input0,gt0,input1,gt1,input2,gt2):
        self.vals = []
        int_loss = criterion_L2(input0,gt0)
        # print(input0.dtype,gt0.dtype)
        dep_loss = criterion_L2(input1,gt1)
        meas_loss = criterion_L2(input2,gt2) 
        # print(input2.max(),gt2.max())
        
        self.vals.append(0*int_loss)
        self.vals.append(0*dep_loss)
        self.vals.append(1*meas_loss)
        
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
    
