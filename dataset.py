import os
import os.path
import numpy as np
import random
import h5py
import torch
#import cv2
from PIL import Image
import glob
import torch.utils.data as udata
from utils import data_augmentation

import torchvision.transforms as transforms

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(root, data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    #scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    n_path = 'data/noise.txt'
    gt_path = 'data/gt.txt'

    #root = '/data0/lichi/denoising/sRGB/SIDD_Medium_Srgb/'
    
    with open(n_path, 'r') as f:
            lines = f.readlines()
            noise_list = [
                i.strip('\n') for i in lines
            ]
            
    with open(gt_path, 'r') as f:
            lines = f.readlines()
            gt_list = [
                i.strip('\n') for i in lines
            ]
            
    h5f = h5py.File('data/train.h5', 'w')
    train_num = 0
    for i in range(len(noise_list)):
        #n_img = cv2.imread(noise_list[i])
        #gt_img = cv2.imread(gt_list[i])

        n_img = Image.open(os.path.join(root, noise_list[i]))
        gt_img = Image.open(os.path.join(root, gt_list[i]))

        # Convert to numpy
        n_img = np.array(n_img, dtype=np.float16)
        gt_img = np.array(gt_img, dtype=np.float16)

        h, w, c = n_img.shape
        for k in range(len(scales)):
            #Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            #n_img = np.expand_dims(n_img[:,:,0].copy(), 0)
            n_img = n_img.transpose(2, 0, 1)
            n_img = np.float16(normalize(n_img))
            n_patches = Im2Patch(n_img, win=patch_size, stride=stride)

            gt_img = gt_img.transpose(2, 0, 1)
            gt_img = np.float16(normalize(gt_img))
            gt_patches = Im2Patch(gt_img, win=patch_size, stride=stride)

            print("file: %s scale %.1f # samples: %d" % (noise_list[i], scales[k], n_patches.shape[3]*aug_times))
            for n in range(n_patches.shape[3]):
                patches = np.concatenate((n_patches[:,:,:,n], gt_patches[:,:,:,n]), axis = 0)
                data = patches.copy()
                #h5f.create_dataset(str(train_num), data=data)
                #train_num += 1
                for m in range(aug_times):
                    data_aug = data
                    #data_aug = data_augmentation(data, np.random.randint(0,8))
                    #data_aug = data_augmentation(data, m+1)
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m), data=data_aug)
                    train_num += 1
                    print(train_num)
    h5f.close()
    '''
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('val set, # samples %d\n' % val_num)
    '''
    print('training set, # samples %d\n' % train_num)



class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.output_size = (48, 48)
        if self.train:
            h5f = h5py.File('data/train.h5', 'r')
        else:
            h5f = h5py.File('data/val_split.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
        
    def preprocess(self, x):
        dh = np.random.randint(1, 200, size=1)
        dw = np.random.randint(1, 200, size=1)
        x = x[:, dh[0]:dh[0]+48, dw[0]:dw[0]+48]    #48, 200
        #mode = np.random.randint(0, 2, size=1)
        #x = data_augmentation(x, mode)
        y = x
        return y
        
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('data/train.h5', 'r')
        else:
            h5f = h5py.File('data/val_split.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        data = self.preprocess(data)
        h5f.close()
        return torch.Tensor(data)

def random_augumentation(v, op=0):

    v2np = v.data.numpy()
    # v2np = v
    if   op == 0:
        tfnp = v2np
    elif op == 1:
        tfnp = v2np[:, :, :, ::-1].copy()
    elif op == 2:
        tfnp = v2np[:, :, ::-1, :].copy()
    elif op == 3:
        tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    ret = torch.Tensor(tfnp)

    return ret
