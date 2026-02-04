import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import scipy.io as scio
import logging
from util.SetRandomSeed import set_seed, worker_init
import util.SetDistTrain as utils
from tqdm import tqdm
from sklearn.metrics import accuracy_score as ACC
 
from metric import RMSE, PSNR, SSIM ,AverageMeter, crop_to_cal,MAD,cal_psnr
from tools import*
import torch.nn.functional as F
from models.network import dy_nlos

from util import dynamic_dataset

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2

def main():
    
    model = dy_nlos(in_ch=1,out_ch=2,ds=8,spatial=128,tlen=256,bin_len=0.01*2,wall_size=2,frames=3,local_rank=-1)
    model_root = '/data/yueli/nlos_video_output/allF24_NDP_v3_new/F3_mamba_S2PDP_150_15F_rsd6_D1_refnlmtv_2024_41/'
    # model_root = '/data/yueli/nlos_video_output/allF24_NDP_v3_com2/F3_mamba_COM2_S2PDP_15F_rsd6_D1_ref_nlm_2024_45/'
    # model_root = '/data/yueli/nlos_video_output/allF24_NDP_v3_com2/F3_mamba_COM2_S2PDP_150_15F_rsd6_D1_ref_nlmtv_2024_49/'
    model_root = '/data/yueli/nlos_video_output/allF24_NDP_v3_com2/F3_VIT_COM2_S2PDP_150_15F_rsd6_D1_ref_nlmtv_2024_49/'
    model_root = '/data/yueli/nlos_video_output/allF24_NDP_v3_com2/F3_VIT_COM2_S2PDP_150_15F_rsd6_D1_ref_nlmtv_woclamp_2024_410/'
    
    all_model_file = []
    model_files = os.listdir(model_root)
    # model_files.sort()
    for fi in model_files:
        if ('pth' in fi) and ('END' not in fi):
            # if 'epoch_49_iter_24400' in  fi: #    epoch_49_iter_40650  epoch_49_iter_24400
            fi_d = os.path.join(model_root, fi)
            all_model_file.append(fi_d)
    
    all_model_file.sort() # = sorted(all_model_file)
    # print(all_model_file)
    if logging.root: del logging.root.handlers[:]
    logging.basicConfig(
      level=logging.INFO,
      handlers=[
        logging.FileHandler(model_root + '/s2pdp_24F_allepochsyn_evluationnew.log' ),
        logging.StreamHandler()
      ],
      format='%(relativeCreated)d:%(levelname)s:%(process)d-%(processName)s: %(message)s'
    )
    model.cuda()
    model = torch.nn.DataParallel(model)
    # print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    # test_filter_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/new_path/test_all_24_pathf.txt'
    # test_total_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/new_path/test_all_24_path.txt'
    test_filter_path = '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/test_24F_pathfnew.txt'
    test_total_path = '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/test_24F_pathnew.txt'
    # test_filter_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/old_path/test_path_filter.txt'
    # test_total_path = '/data/yueli/dataset/NLOS_bike_video_3dhelix_pro/old_path/test_path.txt'
    
    test_data = dynamic_dataset.DynamicNLOSDataset(test_filter_path,test_total_path,512,256,8,128,0.000,True,3,False,24)
    val_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    out_path = model_root + '/syn_new/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    #     # os.makedirs(os.path.join(out_path,'mat'), exist_ok=True)
        os.makedirs(os.path.join(out_path,'int'), exist_ok=True)
    #     os.makedirs(os.path.join(out_path,'dep'), exist_ok=True)
        
    
    cal_ssim = SSIM().cuda()
    cal_mad = MAD().cuda()
    mse = nn.MSELoss()
    
    total_time = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    for i in range(len(all_model_file)):
        l_psnr = []
        l_ssim = []
        l_mad = []
        l_im = []
        l_rmse = []
        acc_sum = []
        checkpoint = torch.load(all_model_file[i], map_location="cpu")
        model_dict = model.state_dict()
        ckpt_dict = checkpoint['state_dict']
        model_dict.update(ckpt_dict)
        model.load_state_dict(model_dict)
        print("Loaded!",all_model_file[i])
        niter = 0
        with torch.no_grad():
            model.eval()
            for sample in tqdm(val_loader):
                ds_meas, pad_meas, raw_meas,dep_gt, img_gt = sample["ds_meas"].type(dtype), sample["pad_meas"].type(dtype), \
                sample["raw_meas"].type(dtype), sample["dep_gt"].type(dtype), sample["img_gt"].type(dtype)
                
                data = {'ds_meas':ds_meas, 'pad_meas':pad_meas, 'raw_meas':raw_meas, 'dep_gt':dep_gt, 'img_gt':img_gt}
                model.module.set_input(data)
                model.module.test_one_iteration()
                rendered_img = model.module.rendered_img
                rendered_depth = model.module.rendered_depth
                dep_gt = model.module.dep_gt
                img_gt = model.module.int_gt
                
                # img_gt = torch.clamp(img_gt, 0, 1)
                mask = dep_gt > 0
                mask = mask.float()
                ###### predict ######
                
                # box_gt, box_pred = crop_to_cal(img_gt,rendered_img)
                box_gt, box_pred = img_gt,rendered_img
                box_gt, box_pred = box_gt.cuda(), box_pred.cuda()
                # box_dep_gt, box_dep_pred = crop_to_cal(dep_gt,rendered_depth)
                box_dep_gt, box_dep_pred = dep_gt,rendered_depth
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
                foreground_gt[foreground_gt <= 20] = 0
                foreground_gt[foreground_gt > 0] = 1
                foreground_pred = box_pred.detach().cpu().numpy() * 255
                foreground_pred[foreground_pred <= 20] = 0
                foreground_pred[foreground_pred > 0] = 1       
                acc_sum.append(ACC(foreground_pred[0,0], foreground_gt[0,0]))

                ###### store ######

                front_view = rendered_img[0].clamp(0,1).cpu().numpy().transpose(1,2,0)
                depth_view = rendered_depth[0].clamp(0,1).cpu().numpy().transpose(1,2,0)
                # # depth_rw = depth_rw[0].cpu().numpy().transpose(1,2,0)

                view_gt = img_gt[0].cpu().numpy().transpose(1,2,0)
                cv2.imwrite(os.path.join(out_path,'int/') + f'{niter}_int.png', (front_view*255))
                cv2.imwrite(os.path.join(out_path,'int/') + f'{niter}_gt.png', (view_gt*255))
                # cv2.imwrite(os.path.join(out_path,'int/') + f'{niter}_dep.png', (depth_view/np.max(depth_view)*255))
                cv2.imwrite(os.path.join(out_path,'int/') + f'{niter}_dep.png', (depth_view*255))
                # # scio.savemat(os.path.join(out_path,'mat/') + f'{niter}.mat',{'dep_pre':depth_view,\
                
                # logging.info("niter:%3d,psnr: %f,ssim: %f,rmse: %f,mad: %f,acc: %f" % (niter, im_psnr.item(),im_ssim.item(),dep_rmse.item(),dep_mad.item(),ACC(foreground_pred[0,0], foreground_gt[0,0])))
                niter += 1

        logging.info(all_model_file[i])
        logging.info("img_ssim: %f" % (float(sum(l_ssim))/float(len(l_ssim))))
        logging.info("dep_mad: %f" % (float(sum(l_mad))/float(len(l_mad))))
        logging.info("img_psnr: %f" % (float(sum(l_psnr))/float(len(l_psnr))))
        logging.info("dep_rmse: %f" % (float(sum(l_rmse))/float(len(l_rmse))))
        logging.info("acc: %f" % (float(sum(acc_sum))/float(len(acc_sum))))

        print('average time is ',total_time/len(val_loader))



if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Sleeping...")
    time.sleep(3600*0)
    print("Wake UP")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




