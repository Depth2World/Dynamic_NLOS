 # The train file for network
# Based on pytorch 1.8
# Use the docker: nlos_trans:1.8 in Ubuntu
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
from util.SetRandomSeed import set_seed, worker_init
from tqdm import tqdm
import torch.nn.functional as F
import imageio


from models.utils_pytorch import phasor_1_10
from models.utils_pytorch import fk_1_10
from models.utils_pytorch import lct_1_10


import cv2
cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
lsmx = torch.nn.LogSoftmax(dim=1)
smx = torch.nn.Softmax(dim=1)
from pro.Loss import criterion_KL, criterion_L2
from skimage.metrics import structural_similarity as ssim

def main():
    
    # baseline   
    
    spatial = 128
    temp_bin = 512
    bin_len = ( 512 // temp_bin ) * 0.0096

    # model = phasor_1_10.phasor(spatial=spatial, crop=temp_bin, bin_len=bin_len, wall_size=1.85, sampling_coeff=3.0, cycles=5,dnum=1)
    # model = fk_1_10.lct_fk(spatial=spatial, crop=temp_bin, bin_len=bin_len, wall_size=1.85, dnum=1)
    model = lct_1_10.lct(spatial=spatial, crop=temp_bin, bin_len=bin_len, wall_size=1.85, method='lct',material='diffuse',dnum=1)
    # model = lct_1_10.lct(spatial=spatial, crop=temp_bin, bin_len=bin_len, wall_size=1.85, method='lct',material='specular',dnum=1)  ## conducted on ours data due to the distance is too small
    
    model.cuda()
    model = torch.nn.DataParallel(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Numbers of parameters are: {}".format(num_params))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    print("Start eval...")
    rw_path  = '/data/yueli/dataset/NLOS_RW/1108_video/'
    out_path = '/data/yueli/nlos_video_output/traditional_methods/lct/1108_video_resize128/'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
        
    for i in range(len(all_file)):
        frame_save_path = out_path + files[i][:-4]
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path, exist_ok=True)
        
        data = scio.loadmat(all_file[i])
        data = data['data']  #sig final_meas measlr   # frames t h w 
        print(data.shape)
        
        video_file = out_path + files[i][:-4] + '.mp4'
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(video_file,fourcc,10,(64,64),True)
        writer = imageio.get_writer(video_file, fps=5)
        
        for index in range(1,data.shape[0]-1):  # not the first and the last frame
            raw_data = data[index,...][None]  # 1 t h w
            # print(raw_data.shape)
            transient_data = np.asarray(raw_data).astype(np.float32)
            transient_data = np.ascontiguousarray(transient_data)
            Mea = torch.from_numpy(transient_data[None])
            print(Mea.mean())
            ds = 4
            if False:
                mask = torch.zeros_like(Mea) # 3 t h w
                for index_i in range(ds//2, 64, ds):
                    for index_j in range(ds//2, 64, ds):
                        mask[:,:,index_i,index_j] = 1
                M_mea =  Mea * mask
            else:
                M_mea = F.interpolate(Mea,[512,128,128])
            
            with torch.no_grad():
                model.eval()
                # print(M_mea.shape)
                re = model(M_mea,[0,0,0],[temp_bin,temp_bin,temp_bin])
                volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
                # zdim = volumn_MxNxN.shape[0] * 100 // 128
                # volumn_MxNxN = volumn_MxNxN[:zdim]
                # print('volumn min, %f' % volumn_MxNxN.min())
                # print('volumn max, %f' % volumn_MxNxN.max())
                volumn_MxNxN[volumn_MxNxN < 0] = 0
                
                front_view = np.max(volumn_MxNxN, axis=0)
                front_view = front_view / np.max(front_view)
                front_view = (front_view*255).astype(np.uint8)
                # depth_view = np.argmax(volumn_MxNxN, axis=0)
                # depth_view = depth_view.astype(np.float32)/np.max(depth_view)
                # depth_view = (depth_view*255).astype(np.uint8)
                import cv2
                cv2.imwrite(frame_save_path + f'/{index}_int.png', (front_view / np.max(front_view))*255)
            
            # out.write(front_view)
            writer.append_data(front_view)
        writer.close()
        # out.release() 
    
if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Sleeping...")
    time.sleep(3600*0)
    print("Wake UP")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




