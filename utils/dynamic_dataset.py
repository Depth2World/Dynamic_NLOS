from enum import EnumMeta
import os 
import glob
from pickle import TRUE
import numpy as np
import cv2
from torch.utils.data import Dataset,DataLoader
import scipy.io as sio
import random
import torch
import torch.nn.functional as F
from math import log2
import sys

def list_file_path(root_path,fortrain=True,shineness=[0]):
    modeldirs = []
    testmodeldirs = []
    mea = []
    im = []
    de =[]
    test_mea = []
    test_im = []
    test_de =[]
    root_path = [root_path]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))      # file_path/0/XXX
            
            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))#[:11]   # file_path/0/XXX/shinexxx_rot xxx
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspadPDP*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    de.append(path)
            for modeldir in modeldir_all[250:]:
                # print(modeldir)
                rotdirs = glob.glob('%s/shin*' % (modeldir))[:1]#[:11]   # file_path/0/XXX/shinexxx_rot xxx
                testmodeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspadPDP*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/all_i*.mat' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/all_d*.mat' % (dir))
                    test_de.append(path)
    
    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': testmodeldirs}
    # import ipdb
    # ipdb.set_trace()
    if fortrain:
        return train_sample
    else:
        return test_sample
    

def list_file_path_bike(root_path,fortrain=TRUE,shineness=[0]):
    modeldirs = []
    test_modeldirs = []
    mea = []
    im = []
    de =[]
    test_mea = []
    test_im = []
    test_de =[]
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))      # file_path/0/XXX
            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))   # file_path/0/XXX/shinexxx_rot xxx
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    de.append(path)
                    
            for modeldir in modeldir_all[250:]:
                # print(modeldir)
                rotdirs = glob.glob('%s/shin*' % (modeldir))#[:1]   # file_path/0/XXX/shinexxx_rot xxx
                test_modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    test_de.append(path)
                    
    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': test_modeldirs}
    # import ipdb
    # ipdb.set_trace()
    if fortrain:
        return train_sample
    else:
        return test_sample
    

    
def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)


def read_dataset(source):
    if not os.path.exists(source):
        print("Setting file %s for dataset doesn't exist." % (source))
        sys.exit()
    else:
        with open(source) as split_f:
            path_samples = split_f.readlines()
        print('{}sequences have been loaded'.format(len(path_samples)))
    return path_samples




class DynamicNLOSDataset(Dataset):
    def __init__(
        self, 
        filter_path,               # dataset root directory
        total_path,               # dataset root directory
        clip=512,           # time range of histograms
        size=256,           # measurement size (unit: px)
        d_s=1,            # scaling factor (float or float tuple)
        target_size=128,    # target image size (unit: px)
        target_noise=0,     # standard deviation of target image noise
        mask = False,
        frame_num = 3,
        training = True,
        length = 50
    ):
        super(DynamicNLOSDataset, self).__init__()
        self.filter_path = read_dataset(filter_path)
        self.total_path = read_dataset(total_path)
        self.clip = clip
        self.size = size
        self.sp_ds = d_s
        self.target_size = target_size
        self.target_noise = target_noise
        self.mask = mask
        self.frame_num = frame_num
        self.training = training
        self.length = length

    def add_poisson(self,x,db=20):
        x_mean = torch.mean(x)
        mag = 10**(db/10)
        rate = x_mean/mag
        poisson_ = torch.distributions.Poisson(rate)
        poisson_noise = poisson_.sample(x.shape)
        
        return x + poisson_noise
        
    

    def _load_raw_meas(self, real_idx):
        # path = self.total_path[real_idx][:-1] + '/video-confocalspadPDPS-gray-full.mp4' # old data
        path = self.total_path[real_idx][:-1] + '/video-confocalspad-gray-full.mp4' # new data
        # path = self.total_path[real_idx][:-1] + '/video-confocalspadR-gray-full.mp4' # new data
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('hdr','mp4')
        try:
            if ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x = x.reshape(-1, x.shape[1], x.shape[1], 3)
                x = x.transpose(3, 0, 1, 2)
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened() 
                ims = []
                # Read until video is completed
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     ## train gray need use
                        ims.append(frame)
                    else:
                        break
                # When everything done, release the video capture object
                cap.release()
                raw = np.array(ims, dtype=np.float32) / 255.0  ## normalize to 0-1
                raw = raw[:,:,:,np.newaxis]
                raw = raw.transpose(3,0,1,2)   # 3 600 256 256  
            if raw.ndim == 3:
                raw = raw[None]
            # clip temporal range
            raw = raw[:, :self.clip] 
            # spatial adjust
            ### resize to 64*64
            sa = self.size//self.target_size
            for _ in range(int(log2(sa))):
                raw = (raw[:,:,0::2,:] + raw[:,:,1::2,:]) / 2
                raw = (raw[:,:,:,0::2] + raw[:,:,:,1::2]) / 2
                
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))
        raw = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
        return raw


    def _load_meas(self, real_idx,db):
        # path = self.total_path[real_idx][:-1] + '/video-confocalspadR-gray-full.mp4' # new data
        if self.training:
            path = self.total_path[real_idx][:-1] + '/video-confocalspadS2-gray-full.mp4' # new data
            # path = self.total_path[real_idx][:-1] + '/video-confocalspadN-gray-full.mp4' # new data
        else:
            path = self.total_path[real_idx][:-1] + '/video-confocalspadS2-gray-full.mp4' # new data
            # path = self.total_path[real_idx][:-1] + '/video-confocalspadN-gray-full.mp4' # new data

        # path = self.total_path[real_idx][:-1] + '/video-confocalspad-gray-full.mp4' # new data
        # path = self.total_path[real_idx][:-1] + '/video-confocalspadPDPS-gray-full.mp4' # old data
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('hdr','mp4')
        try:
            if ext == 'hdr':
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x = x.reshape(-1, x.shape[1], x.shape[1], 3)
                x = x.transpose(3, 0, 1, 2)
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened() 
                ims = []
                # Read until video is completed
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     ## train gray need use
                        ims.append(frame)
                    else:
                        break
                # When everything done, release the video capture object
                cap.release()
                raw = np.array(ims, dtype=np.float32) / 255.0  ## normalize to 0-1
                raw = raw[:,:,:,np.newaxis]
                raw = raw.transpose(3,0,1,2)   # 3 600 256 256  
                
            if raw.ndim == 3:
                raw = raw[None]
            # clip temporal range
            raw = raw[:, :self.clip] 
            # resize temporal
            # raw = (raw[:,::2,:,:] + raw[:,1::2,:,:]) / 2

            # spatial adjust resize to 128*128
            
                ## mode 1  for real-world is worse than mode2 
            # raw = raw[:,:,sa//2::sa,sa//2::sa]
                ## mode 2 
            sa = self.size // (self.target_size)
            for _ in range(int(log2(sa))):
                raw = (raw[:,:,0::2,:] + raw[:,:,1::2,:])
                raw = (raw[:,:,:,0::2] + raw[:,:,:,1::2])
                
        except:
            raise ValueError('measurement loading failed: {:s}'.format(path))
            
        # spatial sub-sampling
        if self.mask:
            raw = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
            mask = torch.zeros_like(raw) # 1 t h w
            for index_i in range(self.sp_ds//2, self.target_size, self.sp_ds):
                for index_j in range(self.sp_ds//2, self.target_size, self.sp_ds):
                    mask[:,:,index_i,index_j] = 1
            pad_meas =  raw * mask
        if self.sp_ds >= 1:
            c, t, h, w = raw.shape
            ### 2024.3.16
            # ds_meas = raw
            # for _ in range(int(log2(self.sp_ds))):
            #     ds_meas = (ds_meas[:,:,0::2,:] + ds_meas[:,:,1::2,:])
            #     # ds_meas = (ds_meas[:,:,:,0::2] + ds_meas[:,:,:,1::2])

            # vertical 
            # ds_meas = ds_meas[:,:,self.sp_ds//2::self.sp_ds,:]
            # horizontal
            # ds_meas = ds_meas[:,:,:,self.sp_ds//2::self.sp_ds]

            # initial    
            ds_meas = raw[:,:,self.sp_ds//2::self.sp_ds,self.sp_ds//2::self.sp_ds]

        # ds_meas = torch.from_numpy(ds_meas.astype(np.float32))              # (1/3, t, h, w)
        # x = F.interpolate(x,size=(self.target_size,) * 2, mode='bicubic') 
        # pad_meas = torch.from_numpy(pad_meas.astype(np.float32))              # (1/3, t, h, w)
        # raw = torch.from_numpy(raw.astype(np.float32))              # (1/3, t, h, w)
        # print(ds_meas.shape)
        # return x/x.max(), raw/raw.max()
        return ds_meas, pad_meas, raw

    def _load_image(self, real_idx):
        path = self.total_path[real_idx][:-1] + '/all_img.mat'
        # path = self.total_path[real_idx][:-1] + '/all_img_rsdgt.mat'
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'png')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['img']     # 1 256 256
                # print(x.shape)
                x = x.transpose(1,2,0)
                x = cv2.resize(x, (self.target_size, self.target_size))          # (h, w)
                x = x[np.newaxis,:,:]  # 1 64 64
                x = x / np.max(x)  
                # print(x.shape)
                
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x = x / 255
        except:
            raise ValueError('image loading failed: {:s}'.format(path))
       
        x = torch.from_numpy(x.astype(np.float32))              # (c, h, w)
        if self.target_noise > 0:
            x += torch.randn_like(x) * self.target_noise
            x = torch.clamp(x, min=0, max=1)

        return x

    def _load_depth(self, real_idx):
        path = self.total_path[real_idx][:-1] + '/all_depth.mat'
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'png')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['depth']   # 1 256 256 3 
                x = x[...,0]
                x = x.transpose(1,2,0)
                x = cv2.resize(x, (self.target_size, self.target_size))          # (h, w)
                x = x.clip(0,1)  ## note here ! 
                x = x[np.newaxis,:,:]
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = x[..., 0]
                x = x.reshape(-1, x.shape[1], x.shape[1])
                x = x.transpose(1, 2, 0)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))
        x = torch.from_numpy(x.astype(np.float32))              # (1, h, w)
        return x
    
    def load_all_data(self,real_idx):
        ds_meas, pad_meas, raw_meas, ints, deps = [], [], [], [], []
        # for i in [-1,0,1]:
        db = 10 #np.random.randint(10, 20) # 5-20
        for i in range(-1,2):
            t_meads, t_meapad, t_raw_mea = self._load_meas(real_idx+i,db)
            # print('init x', t_mea.max(),'init raw', t_raw_mea.max())
            # print(t_meads.shape,t_meapad.shape,t_raw_mea.shape)
            # t_raw_mea = self._load_raw_meas(real_idx+i)
            intensity = self._load_image(real_idx+i)
            depth = self._load_depth(real_idx+i)
            # print('t_mea', t_mea.max(),t_mea.min())
            # print('t_raw_mea', t_raw_mea.max(),t_raw_mea.min())
            # print('intensity', intensity.max(),intensity.min())
            # print('depth', depth.max(),depth.min())
            # print(t_mea.shape,t_raw_mea.shape,intensity.shape,depth.shape)
            ds_meas.append(t_meads)
            pad_meas.append(t_meapad)
            raw_meas.append(t_raw_mea)
            ints.append(intensity)
            deps.append(depth)
        # print(np.array(ds_meas).shape,np.array(pad_meas).shape,np.array(raw_meas).shape)
        ds_meas, pad_meas, raw_meas= np.stack(ds_meas,axis=0), np.stack(pad_meas,axis=0), np.stack(raw_meas,axis=0) # frames,c,t,h,w
        ints, deps =  np.stack(ints,axis=0), np.stack(deps,axis=0) # frames,c,h,w
        
        # print(meas.shape, raw_meas.shape, ints.shape, deps.shape) 
        return ds_meas, pad_meas, raw_meas, ints, deps
    
    def __len__(self):
        return len(self.filter_path)

    def __getitem__(self, idx):
        # real_idx = (idx // 30) * 32 + idx % 30 + 1
        # real_idx = (idx // 62) * 64 + idx % 62 + 1
        # real_idx = (idx // 48) * 50 + idx % 48 + 1  # old data
        real_idx = (idx // (self.length-2)) * self.length + idx % (self.length-2) + 1  # old data
        # print(str(self.filter_path[idx]))
        # print(str(self.total_path[real_idx]))
        meas, pad_meas, raw_meas, ints, deps = self.load_all_data(real_idx)
        sample = {'ds_meas': meas, 'pad_meas': pad_meas, 'raw_meas': raw_meas, 'dep_gt': deps, 'img_gt': ints}
        return sample



if __name__ == '__main__': 
  
    filter_path = '/data2/yueli/code/nlos_video_eccv2024/util/test_path_filter.txt'
    total_path = '/data2/yueli/code/nlos_video_eccv2024/util/test_path.txt'
    
    train_data = DynamicNLOSDataset(filter_path,total_path,512,256,8,64,0.000,False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1) # drop_last would also influence the performance
    print(len(train_data))
    for index,data in enumerate(train_loader):
        print(data['ds_meas'].shape,data['raw_meas'].shape,data['dep_gt'].shape,data['img_gt'].shape)   
        
        # LFE          torch.Size([4, 1, 512, 16, 16]) torch.Size([4, 1, 512, 128, 128]) torch.Size([4, 1, 1, 128, 128]) torch.Size([4, 1, 1, 128, 128])
        # NLOSDataset  torch.Size([4, 1, 512, 16, 16]) torch.Size([4, 1, 512, 128, 128]) torch.Size([4, 26, 1, 128, 128]) torch.Size([4, 26, 1, 128, 128])