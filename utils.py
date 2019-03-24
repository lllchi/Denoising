import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    # #out = np.transpose(image, (1,2,0))
    # out = image
    # if mode == 0:
    #     # original
    #     out = out
    # elif mode == 1:
    #     # flip up and down
    #     out = np.flipud(out)
    # elif mode == 2:
    #     # rotate counterwise 90 degree
    #     out = np.rot90(out)
    # elif mode == 3:
    #     # rotate 90 degree and flip up and down
    #     out = np.rot90(out)
    #     out = np.flipud(out)
    # elif mode == 4:
    #     # rotate 180 degree
    #     out = np.rot90(out, k=2)
    # elif mode == 5:
    #     # rotate 180 degree and flip
    #     out = np.rot90(out, k=2)
    #     out = np.flipud(out)
    # elif mode == 6:
    #     # rotate 270 degree
    #     out = np.rot90(out, k=3)
    # elif mode == 7:
    #     # rotate 270 degree and flip
    #     out = np.rot90(out, k=3)
    #     out = np.flipud(out)
    # #return np.transpose(out, (2,0,1))
    # return out

    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image, k=2)

    return out

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

def update_lr(ori_lr, epoch):
    #current_lr = ori_lr * (1 - epoch/40.0)**0.9
    current_lr = ori_lr * (0.1 ** ((epoch) // 40))
    return current_lr

