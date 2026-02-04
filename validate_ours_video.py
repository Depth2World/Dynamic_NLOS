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
import imageio
import torch.nn.functional as F
import cv2
import util.SetDistTrain as utils
from tqdm import tqdm

cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim
from models.network import dy_nlos
# from models.network_old import dy_nlos
# from models.network_mea import dy_nlos

def main():
    
    # baseline   
    model = dy_nlos(in_ch=1,out_ch=2,ds=8,spatial=128,tlen=256,bin_len=0.0096*2,wall_size=1.3,frames=3,local_rank=-1)

    model_root = '/data/yueli/nlos_video_output/v3/NPDP_12F_SR_PAD_LCT_D1_REF_2024_316/'
    model_root = '/data/yueli/nlos_video_output/v3/NPDP_12F_SR_LCT_D1_REF_2024_316/'
    model_root = '/data/yueli/nlos_video_output/v3_unalias/NPDP_12F_SR_PAD_LCT_D1_REF_2024_317/'
    out_root = model_root + '/1108_video/'
    
    all_model_file = []
    all_output = []
    model_files = os.listdir(model_root)
    for fi in model_files:
        if 'pth' in fi:
            fi_d = os.path.join(model_root, fi)
            all_model_file.append(fi_d)
            all_output.append(os.path.join(out_root, fi[:-4]))

    model.cuda()
    model = torch.nn.DataParallel(model)
    print("Start eval...")
    rw_path  = '/data/yueli/dataset/NLOS_RW/1108_video/'
    
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
        
    for k in range(len(all_model_file)):
        model_path = all_model_file[k]
        each_out_path = all_output[k]

        if os.path.exists(each_out_path):
            print('Pass', model_path)
            continue
        else:
            print(model_path)
            
        checkpoint = torch.load(model_path, map_location="cpu")
        # model_dict = model.state_dict()
        ckpt_dict = checkpoint['state_dict']
        # model_dict.update(ckpt_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(ckpt_dict)
        
        for i in range(len(all_file)):
            
            out_path = each_out_path + '/' + files[i][:-4]
            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)

            data = scio.loadmat(all_file[i])
            data = data['data']  #sig final_meas measlr 
            data[data<0]=0
            # print(data.max()) 
            video_file = out_path + '.mp4'
            writer = imageio.get_writer(video_file, fps=5)
            for index in range(data.shape[0]-2):  # not the first and the last frame
                # print(index)
                raw_data = data[index:index+3,...]  # 3 t h w
                raw_data[:,:100,:,:] = 0
                max_ = raw_data.max(axis=(-3,-2,-1))
                max_ = np.tile(max_.reshape(3,1,1,1),[1,512,16,16])
                raw_data = raw_data/max_
                # print(raw_data.shape)
                transient_data = np.asarray(raw_data).astype(np.float32)
                transient_data = np.ascontiguousarray(transient_data)
                # Mea = torch.from_numpy(transient_data)
                # Mea = F.interpolate(Mea,size=(32,) * 2, mode='bicubic') 
                # Mea = Mea[None]
                Mea = torch.from_numpy(transient_data[None])
                ds = 8
                if True:
                    new_data = torch.zeros(Mea.shape[0],Mea.shape[1],Mea.shape[2],ds*Mea.shape[3],ds*Mea.shape[4])
                    for index_i in range(0, 16, 1):
                        for index_j in range(0, 16, 1):
                            new_data[:,:,:, ds//2 + ds*index_i,ds//2 + ds*index_j] = Mea[:,:,:,index_i,index_j]
                    pad_Mea =  new_data
                    ds_Mea = Mea

                with torch.no_grad():
                    model.eval()
                    ds_Mea = ds_Mea.unsqueeze(2).cuda()#.to(local_rank)
                    pad_Mea = pad_Mea.unsqueeze(2).cuda()#.to(local_rank)
                    # print(M_mea.shape)
                    # starter.record()
                    model.module.evluation_rw(ds_Mea,pad_Mea)
                    # ender.record()
                    # torch.cuda.synchronize()
                    # curr_time = starter.elapsed_time(ender) # 计算时间
                    # print(curr_time)
                    vlo, im_re, dep_re = model.module.vlo, model.module.rendered_img, model.module.rendered_depth 
                    front_view = im_re.detach().cpu().numpy()[0, 0]
                    front_view = (front_view / np.max(front_view))*255
                    # print(front_view.shape)
                    cv2.imwrite(out_path + f'/{index}_int.png', front_view)  
                writer.append_data(np.clip(front_view,0,255).astype('uint8'))
            writer.close()
            
if __name__=="__main__":
    main()




