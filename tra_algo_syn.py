import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from datetime import datetime
import scipy.io as scio

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import dynamic_dataset

from models.utils_pytorch import phasor_1_10
from models.utils_pytorch import fk_1_10
from models.utils_pytorch import lct_1_10
import torch.nn.functional as F

from metric import RMSE, PSNR, SSIM ,AverageMeter, crop_to_cal, MAD, cal_psnr

import cv2
cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score as ACC
import glob

def main():
    
    # baseline   
    
    spatial = 128
    temp_bin = 512
    bin_len = ( 512 // temp_bin ) * 0.01
    # model = phasor_1_10.phasor(spatial=spatial, crop=temp_bin, bin_len=bin_len, wall_size=2, sampling_coeff=6.0, cycles=5,dnum=1)
    # model = fk_1_10.lct_fk(spatial=spatial, crop=temp_bin, wall_size=2, bin_len=bin_len,dnum=1)
    model = lct_1_10.lct(spatial=spatial, crop=temp_bin, wall_size=2,bin_len=bin_len,method='bp',dnum=1)
    
    model.cuda()
    model = torch.nn.DataParallel(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    # test_filter_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/test_path_filter.txt'
    # test_total_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/test_path.txt'
    # test_filter_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/new_path/test_all_24_pathf.txt'
    # test_total_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/new_path/test_all_24_path.txt'
    test_filter_path = '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/test_24F_pathfnew.txt'
    test_total_path = '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/test_24F_pathnew.txt'
    # val_data = dynamic_dataset.DynamicNLOSDataset(test_filter_path,test_total_path,512,256,1,64,0.000,False)
    val_data = dynamic_dataset.DynamicNLOSDataset(test_filter_path,test_total_path,512,256,8,128,0.000,True,3,False,24)

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    rmse = RMSE().cuda()
    psnr = PSNR().cuda()
    ssim = SSIM().cuda()
    metric_list = ['rmse', 'psnr', 'ssim','acc']
    intensity_metrics = {k: AverageMeter() for k in metric_list}
    
    out_path = f'/data/yueli/nlos_video_output/allF24_NDP_v3_com2/tra_syn_test_new/rsd6_dep512/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    i=0
    
    l_psnr = []
    l_ssim = []
    l_mad = []
    l_im = []
    l_rmse = []
    acc_sum = []

    cal_ssim = SSIM().cuda()
    cal_mad = MAD().cuda()
    mse = nn.MSELoss()
    
    total_time = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        model.eval()
        for sample in tqdm(val_loader):
            ds_meas, pad_meas, raw_meas,dep_gt, img_gt = sample["ds_meas"].type(dtype), sample["pad_meas"].type(dtype), \
                sample["raw_meas"].type(dtype), sample["dep_gt"].type(dtype), sample["img_gt"].type(dtype)
                
            M_mea, raw_mea, dep_gt, img_gt = ds_meas[:,1,...], raw_meas[:,1,...], dep_gt[:,1,...], img_gt[:,1,...]

            # print(raw_mea.mean())
            # print(M_mea.shape, raw_mea.shape, dep_gt.shape, img_gt.shape)
            # print(M_mea.shape)
            M_mea = F.interpolate(M_mea[0,...],size=(128,) * 2, mode='bicubic') 
            M_mea = M_mea[None]

            starter.record()
            re = model(M_mea,[0,0,0],[temp_bin,temp_bin,temp_bin])
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 计算时间
            total_time += curr_time
            
            
            volumn_MxNxN = re.detach().cpu().numpy()[0, 0]
            print(volumn_MxNxN.shape)
            volumn_MxNxN[volumn_MxNxN < 0] = 0
            front_view = np.max(volumn_MxNxN, axis=0)
            front_view = front_view / np.max(front_view)
            ### metirc
            int_pre = torch.from_numpy(front_view[None][None]).cuda()
            
            ### save 
            depth_view = np.argmax(volumn_MxNxN, axis=0)
            depth_view = depth_view.astype(np.float32)/512
            dep_pre = torch.from_numpy(depth_view[None][None]).cuda()

            # box_gt, box_pred = crop_to_cal(img_gt,int_pre)
            box_gt, box_pred = img_gt,int_pre
            box_gt, box_pred = box_gt.cuda(), box_pred.cuda()
            # box_dep_gt, box_dep_pred = crop_to_cal(dep_gt,dep_pre)
            box_dep_gt, box_dep_pred = dep_gt,dep_pre
            box_dep_gt, box_dep_pred = box_dep_gt.cuda(), box_dep_pred.cuda()

            im_psnr = cal_psnr(box_pred, box_gt)
            l_psnr.append(im_psnr.item())

            im_ssim = cal_ssim(box_pred, box_gt)
            l_ssim.append(im_ssim.item())
            
            dep_mad = cal_mad(box_dep_pred, box_dep_gt)
            l_mad.append(dep_mad.item())
            
            dep_rmse = torch.sqrt(mse(box_dep_pred, box_dep_gt))
            l_rmse.append(dep_rmse.item())

            foreground_gt = box_gt.detach().cpu().numpy() * 255
            foreground_gt[foreground_gt < 20] = 0
            foreground_gt[foreground_gt > 0] = 1
            foreground_pred = box_pred.detach().cpu().numpy() * 255
            foreground_pred[foreground_pred < 20] = 0
            foreground_pred[foreground_pred > 0] = 1       
            acc_sum.append(ACC(foreground_pred[0,0], foreground_gt[0,0]))
            

            front_view = (front_view*255).astype(np.uint8)
            depth_view = (depth_view*255).astype(np.uint8)
            cv2.imwrite(out_path +  f'{i}_int.png',front_view)
            cv2.imwrite(out_path + f'{i}_dep.png',depth_view)

            # int_gt = img_gt.detach().cpu().numpy()[0, 0]
            # gt_view = (int_gt*255).astype(np.uint8)
            # cv2.imwrite(out_path +  f'{i}_gt.png',gt_view)

            ###gt 
            # out_path_gt = '/data/yueli/nlos_sp_output/nips_cs/traditional_algos/unseensyn/gt/'
            
            i+=1
            
    print("img_ssim: %f" % (float(sum(l_ssim))/float(len(l_ssim))))
    print("dep_mad: %f" % (float(sum(l_mad))/float(len(l_mad))))
    print("img_psnr: %f" % (float(sum(l_psnr))/float(len(l_psnr))))
    print("dep_rmse: %f" % (float(sum(l_rmse))/float(len(l_rmse))))
    print("acc_sum: %f" % (float(sum(acc_sum))/float(len(acc_sum))))
    
if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Sleeping...")
    time.sleep(3600*0)
    print("Wake UP")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




