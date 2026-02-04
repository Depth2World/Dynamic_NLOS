import torch
import logging
from .worker import worker
from util import dynamic_dataset
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, RandomSampler

import imageio
from models.network import dy_nlos
# from models.network_old import dy_nlos
# from models.network_mea import dy_nlos
from pro.Loss import criterion_KL, criterion_L2
from metric import RMSE, PSNR, SSIM, AverageMeter
from sklearn.metrics import accuracy_score as ACC
import os 
import scipy.io as scio
import numpy as np
import torch.nn.functional as F
import cv2
###DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
metric_list = ['rmse', 'psnr', 'ssim','acc']
val_metrics = {k: AverageMeter() for k in metric_list}


class dy_model(worker):
    def __init__(self,args):
        super().__init__(args=args)
       
        
    def load_resume(self):
        
        if self.args.resume:
            if os.path.exists(self.args.resmod_dir):
                logging.info("Loading checkpoint from {}".format(self.args.resmod_dir))
                checkpoint = torch.load(self.args.resmod_dir, map_location="cpu")
                # load start epoch
                try:
                    self.start_epoch = checkpoint['epoch']
                    logging.info("Loaded and update start epoch: {}".format(self.start_epoch))
                except KeyError as ke:
                    self.start_epoch = 0
                    logging.info("No epcoh info found in the checkpoint, start epoch from 1")
                # load iter number
                try:
                    self.n_iter = checkpoint["n_iter"]
                    logging.info("Loaded and update start iter: {}".format(self.n_iter))
                except KeyError as ke:
                    self.n_iter = 0
                    logging.info("No iter number found in the checkpoint, start iter from 0")

                # load learning rate
                try:
                    self.args.lr_rate = checkpoint["lr"]
                except KeyError as ke:
                    logging.info("No learning rate info found in the checkpoint, use initial learning rate:")
                # load model params
                # model_dict = model.state_dict()
                try:
                    ckpt_dict = checkpoint['state_dict']
                    self.model.load_state_dict(ckpt_dict)
                    logging.info("Loaded and update model states!")
                except KeyError as ke:
                    logging.info("No model states found!")
                    sys.exit("NO MODEL STATES")

                logging.info("Checkpoint load complete!!!")

            else:
                logging.info("No checkPoint found at {}!!!".format(self.args.resmod_dir))
                sys.exit("NO FOUND CHECKPOINT ERROR!")
        else:
            logging.info("Do not resume! Use initial params and train from scratch.")

    def init_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.opter == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=self.args.lr_rate, weight_decay=self.args.weit_decay)
        else:
            self.optimizer = torch.optim.Adam(params, lr=self.args.lr_rate)

    def train_model(self):
        train_filter_path = self.args.train_filter_path
        train_total_path = self.args.train_total_path
        train_data = dynamic_dataset.DynamicNLOSDataset(train_filter_path,train_total_path,512,256,8,128,0.000,self.args.mask,3,True,15)
        train_sampler = DistributedSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler,batch_size=self.args.train_bacth_size, num_workers=self.args.num_workers, pin_memory=True, drop_last=True)
        
        test_filter_path = self.args.test_filter_path
        test_total_path = self.args.test_total_path
        test_data = dynamic_dataset.DynamicNLOSDataset(test_filter_path,test_total_path,512,256,8,128,0.000,self.args.mask,3,False,24)
        val_loader = DataLoader(test_data,batch_size=self.args.test_bacth_size, num_workers=self.args.num_workers, pin_memory=True)
        
                
        self.model = dy_nlos(in_ch=1,out_ch=2,ds=8,spatial=128,tlen=256,bin_len=0.02,wall_size=2,frames=3,local_rank=self.local_rank)
        ### load resume
        self.model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank],find_unused_parameters=True)
        self.load_resume()
        
        logging.info("Models constructed complete! Paralleled on {} GPUs".format(torch.cuda.device_count()))
        
        # optimizer 
        self.init_optimizer()

        for epoch in range(self.start_epoch,self.args.num_epoch):
            self.current_epoch = epoch
            
            if True:
                self.model.train()
                train_loader.sampler.set_epoch(epoch)
                logging.info('{} epcoh start training!'.format(epoch))
                # self.train_new_epoch()
                loss_rsd = loss_mea = loss_int = loss_dep = loss_tv = 0

                ### training
                for iteration, sequence in enumerate(train_loader):
                    # print('training',self.n_iter)
                    # if self.n_iter == 1:
                        # break
                    self.n_iter += 1
                    data = self.prepare_data(sequence)
                    self.model.module.set_input(data)
                    
                    self.model.module.train_one_iteration(self.optimizer)
                    # self.loss = self.model
                    errs = self.model.module.get_loss()
                    
                    loss_rsd += errs[0].item()
                    loss_mea += errs[1].item()
                    loss_int += errs[2].item()
                    loss_dep += errs[3].item()
                    loss_tv += errs[4].item()
                
                if dist.get_rank() == 0:
                    loss_rsd /= (iteration+1)
                    loss_mea /= (iteration+1)
                    loss_int /= (iteration+1)
                    loss_dep /= (iteration+1)
                    loss_tv /= (iteration+1)
                    self.writer.add_scalar('Training/Loss_rsd', loss_rsd, epoch)
                    self.writer.add_scalar('Training/Loss_mea', loss_mea, epoch)
                    self.writer.add_scalar('Training/Loss_int', loss_int, epoch)
                    self.writer.add_scalar('Training/Loss_dep', loss_dep, epoch)
                    self.writer.add_scalar('Training/Loss_tv', loss_tv, epoch)

                # if dist.get_rank() == 0:
                #     evluation_realworld_fk(self.local_rank,self.model.state_dict(),self.args.model_dir,self.current_epoch,self.n_iter,self.args.mask) 
                #     evluation_realworld(self.local_rank,self.model.state_dict(),self.args.model_dir,self.current_epoch,self.n_iter,self.args.mask) 
            if True:    
                logging.info('{} epcoh start validation!'.format(epoch))
                ## validation
                loss_rsd_test = loss_mea_test = loss_int_test = loss_dep_test = loss_tv_test = 0
                rmse = RMSE().cuda()
                psnr = PSNR().cuda()
                ssim = SSIM().cuda()
                self.model.eval()
                with torch.no_grad():
                    for iters, seq in enumerate(val_loader):
                        # print('test',iters)
                        # if iters ==1 :
                            # break
                        data = self.prepare_data(seq)
                        self.model.module.set_input(data)
                        self.model.module.test_one_iteration()
                        errs = self.model.module.get_loss()
                        
                        loss_rsd_test += errs[0].item()
                        loss_mea_test += errs[1].item()
                        loss_int_test += errs[2].item()
                        loss_dep_test += errs[3].item()
                        loss_tv_test += errs[4].item()


                        foreground_gt = self.model.module.int_gt.detach().cpu().numpy() * 255
                        foreground_gt[foreground_gt < 10] = 0
                        foreground_gt[foreground_gt > 0] = 1
                        foreground_pred = self.model.module.rendered_img.detach().cpu().numpy() * 255
                        foreground_pred[foreground_pred < 10] = 0
                        foreground_pred[foreground_pred > 0] = 1       
                        acc_sum = 0
                        for i in range(self.model.module.rendered_img.shape[0]):
                            # print(foreground_pred[i,0].shape,foreground_gt[i,0].shape)
                            # acc_sum+=ACC(foreground_pred[i,0], foreground_gt[i,0])
                            acc_sum+=ACC(foreground_gt[i,0], foreground_gt[i,0])
                            # print(ACC(foreground_gt[i,0], foreground_gt[i,0]))
                        acc_sum/=self.model.module.rendered_img.shape[0]
                        
                        metric_dict = {
                            'rmse': rmse(self.model.module.rendered_img, self.model.module.int_gt).cpu(),
                            'psnr': psnr(self.model.module.rendered_img, self.model.module.int_gt).cpu(),
                            'ssim': ssim(self.model.module.rendered_img, self.model.module.int_gt).cpu(),
                            'acc': ssim(self.model.module.rendered_img, self.model.module.int_gt).cpu(),
                            }
                        for k in metric_list:
                            val_metrics[k].update(metric_dict[k].item())

                if dist.get_rank() == 0:
                    loss_rsd_test /= (iters+1)
                    loss_mea_test /= (iters+1)
                    loss_int_test /= (iters+1)
                    loss_dep_test /= (iters+1)
                    loss_tv_test /= (iters+1)

                    self.writer.add_scalar('Testing/Loss_mea', loss_mea_test, epoch)
                    self.writer.add_scalar('Testing/Loss_rsd', loss_rsd_test, epoch)
                    self.writer.add_scalar('Testing/Loss_int', loss_int_test, epoch)
                    self.writer.add_scalar('Testing/Loss_dep', loss_dep_test, epoch)
                    self.writer.add_scalar('Testing/Loss_tv', loss_tv_test, epoch)

                    for k in metric_list:
                        self.writer.add_scalars(k, {'test': val_metrics[k].item()}, epoch)

                    self.writer.add_images("Testing/inten", self.model.module.rendered_img.clamp(0,1), epoch,dataformats="NCHW")
                    self.writer.add_images("Testing/gt", self.model.module.int_gt, epoch,dataformats="NCHW")
            
             ### save checkpoint
            file_path = self.args.model_dir+"/epoch_{}_iter_{}.pth".format(epoch, self.n_iter)
            self.save_checkpoint(self.n_iter, epoch, self.model, self.optimizer, file_path) 
                
    def prepare_data(self,sequence):
        ds_meas = sequence['ds_meas'].to(self.local_rank)
        pad_meas = sequence['pad_meas'].to(self.local_rank)
        raw_meas = sequence['raw_meas'].to(self.local_rank)
        dep_gt = sequence['dep_gt'].to(self.local_rank)
        img_gt = sequence['img_gt'].to(self.local_rank)
        
        data = {'ds_meas':ds_meas, 'pad_meas':pad_meas, 'raw_meas':raw_meas, 'dep_gt':dep_gt, 'img_gt':img_gt}
        return data



            
def evluation_realworld_fk(local_rank,model_dict,model_dir,current_epoch,n_iter,mask):
    rw_model = dy_nlos(in_ch=1,out_ch=2,ds=4,spatial=64,tlen=1024,bin_len=0.0096,wall_size=2,frames=3,local_rank=local_rank)
    rw_model = torch.nn.DataParallel(rw_model).cuda()#.to(local_rank)
    try:
        rw_model.load_state_dict(model_dict)
        logging.info('Succeed loading statedict for real-world FK testing!')
    except:
        logging.info('Failed loading statedict for real-world FK testing!')
    try:
        rw_path  = '/data/yueli/dataset/NLOS_RW/fk_video/'
        if not os.path.exists(rw_path):
            logging.info('FK real-world data doesnot exist!')
    except:
        raise ValueError('FK measurement loading failed: {:s}'.format(rw_path))
    
    out_path = model_dir + '/fk_video/' + f'epoch_{current_epoch}_iter_{n_iter}/'
    # print(out_path)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path, exist_ok=True)
        
    all_file = []
    files = os.listdir(rw_path)
    for fi in files:
        fi_d = os.path.join(rw_path, fi)
        all_file.append(fi_d)
        
    for i in range(len(all_file)):
        frame_save_path = out_path + files[i][:-4]
        # print(frame_save_path)
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path, exist_ok=True)
        
        data = scio.loadmat(all_file[i])
        data = data['data']  #sig final_meas measlr  # frames t h w 
        # print(data.shape)
        if data.shape[2] == 32:
            data = data[:,:,1::2,1::2]
        elif data.shape[2] == 64:
            data = data[:,:,2::4,2::4]
        else:
            logging.info('FK data size error! ')

        video_file = out_path + files[i][:-4] + '.mp4'
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(video_file,fourcc,10,(64,64),True)
        writer = imageio.get_writer(video_file, fps=5)
        
        for index in range(data.shape[0]-2):  # not the first and the last frame
            # print(index)
            raw_data = data[index:index+3,...]  # 3 t h w
            # print(raw_data.shape)
            transient_data = np.asarray(raw_data).astype(np.float32)
            transient_data = np.ascontiguousarray(transient_data)
            Mea = torch.from_numpy(transient_data[None])
            # print(Mea.dtype)
            # print(Mea.shape) # 1 3 t h w
            ds = 4
            if mask:
                new_data = torch.zeros(Mea.shape[0],Mea.shape[1],Mea.shape[2],ds*Mea.shape[3],ds*Mea.shape[4])
                for index_i in range(0, 16, 1):
                    for index_j in range(0, 16, 1):
                        new_data[:,:,:, ds//2 + ds*index_i,ds//2 + ds*index_j] = Mea[:,:,:,index_i,index_j]
                M_mea =  new_data
            else:
                M_mea = Mea
                # M_mea = F.interpolate(Mea,[1024,64,64], mode='bicubic')
            
            with torch.no_grad():
                rw_model.eval()
                M_mea = M_mea.unsqueeze(2).cuda()#.to(local_rank)
                rw_model.module.evluation_rw(M_mea)
                vlo, im_re, dep_re = rw_model.module.vlo, rw_model.module.rendered_img, rw_model.module.rendered_depth 
                front_view = im_re.detach().cpu().numpy()[0, 0]
                front_view = (front_view / np.max(front_view))*255
                # print(front_view.shape)
                if dist.get_rank() == 0:
                    cv2.imwrite(frame_save_path + f'/{index}_int.png', front_view)  
            writer.append_data(front_view.astype('uint8'))
        writer.close()
        
def evluation_realworld(local_rank,model_dict,model_dir,current_epoch,n_iter,mask):
    rw_model = dy_nlos(in_ch=1,out_ch=2,ds=4,spatial=64,tlen=512,bin_len=0.0096,wall_size=1.85,frames=3,local_rank=local_rank)
    rw_model = torch.nn.DataParallel(rw_model).cuda()#.to(local_rank)
    try:
        rw_model.load_state_dict(model_dict)
        logging.info('Succeed loading statedict for real-world ours testing!')
    except:
        logging.info('Failed loading statedict for real-world ours testing!')
    try:
        rw_path  = '/data/yueli/dataset/NLOS_RW/1108_video/'
        if not os.path.exists(rw_path):
            logging.info('ours real-world data doesnot exist!')
    except:
        raise ValueError('ours measurement loading failed: {:s}'.format(rw_path))
    
    out_path = model_dir + '/1108_video/' + f'epoch_{current_epoch}_iter_{n_iter}/'
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path, exist_ok=True)
        
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
        video_file = out_path + files[i][:-4] + '.mp4'
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(video_file,fourcc,10,(64,64),True)
        writer = imageio.get_writer(video_file, fps=5)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        for index in range(data.shape[0]-2):  # not the first and the last frame
            # print(index)
            raw_data = data[index:index+3,...]  # 3 t h w
            # print(raw_data.shape)
            transient_data = np.asarray(raw_data).astype(np.float32)
            transient_data = np.ascontiguousarray(transient_data)
            Mea = torch.from_numpy(transient_data[None])
            ds = 4
            if mask:
                new_data = torch.zeros(Mea.shape[0],Mea.shape[1],Mea.shape[2],ds*Mea.shape[3],ds*Mea.shape[4])
                for index_i in range(0, 16, 1):
                    for index_j in range(0, 16, 1):
                        new_data[:,:,:, ds//2 + ds*index_i,ds//2 + ds*index_j] = Mea[:,:,:,index_i,index_j]
                M_mea =  new_data
            else:
                M_mea = Mea
                # M_mea = F.interpolate(Mea,[512,64,64], mode='bicubic')
            
            with torch.no_grad():
                rw_model.eval()
                M_mea = M_mea.unsqueeze(2).cuda()#.to(local_rank)
                # print(M_mea.shape)
                # starter.record()
                rw_model.module.evluation_rw(M_mea)
                # ender.record()
                # torch.cuda.synchronize()
                # curr_time = starter.elapsed_time(ender) # 计算时间
                # print(curr_time)
                vlo, im_re, dep_re = rw_model.module.vlo, rw_model.module.rendered_img, rw_model.module.rendered_depth 
                front_view = im_re.detach().cpu().numpy()[0, 0]
                front_view = (front_view / np.max(front_view))*255
                # print(front_view.shape)
                if dist.get_rank() == 0:
                    cv2.imwrite(frame_save_path + f'/{index}_int.png', front_view)  
            writer.append_data(front_view.astype('uint8'))
        writer.close()
        

