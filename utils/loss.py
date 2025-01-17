import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pytorch_msssim import ms_ssim as ssim


def gumbel_softmax_sample(logits, temperature, gumbel, dim):
    '''mip loss'''
    y = logits + gumbel
    return F.softmax(y / temperature, dim)


def cal_snr(noise_img, clean_img):
    noise_img, clean_img = noise_img.detach().cpu().numpy(), clean_img.detach().cpu().numpy()
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal**2
    clean_signal_2 = clean_signal**2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20*math.log10(math.sqrt(sum1)/math.sqrt(sum2))
    return snrr


class MIPloss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.temp = options.loss.temp
        self.lamda = options.loss.lamda
        self.num_slice = options.use_slice
        self.mse = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()

    def reset_gumbel(self, img_fake):
        U = torch.rand_like(img_fake)
        # U = torch.rand(img_fake.size()).cuda()
        self.gumbel = -torch.log(-torch.log(U + 1e-20) + 1e-20)  # sample_gumbel

    def forward(self, img_fake, batch):
        self.reset_gumbel(img_fake)
        target = batch['3D-TOF-MRA']
        pred_mips_c1 = torch.zeros_like(img_fake)
        target_mips_c1 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, :idx+1], self.temp, self.gumbel[:, :idx+1], dim=1)
            # pred_mip = torch.softmax(img_fake[:, :idx+1]/temp, dim=1)
            target_mips_c1[:, idx] = torch.max(target[:, :idx+1], dim=1)[0]
            pred_mips_c1[:, idx] = torch.sum(pred_mip*img_fake[:, :idx+1], dim=1)

        pred_mips_c2 = torch.zeros_like(img_fake)
        target_mips_c2 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, self.num_slice-idx-1:], self.temp, self.gumbel[:, self.num_slice-idx-1:], dim=1)
            # pred_mip = torch.softmax(img_fake[:, :idx+1]/temp, dim=1)
            target_mips_c2[:, idx] = torch.max(target[:, self.num_slice-idx-1:], dim=1)[0]
            pred_mips_c2[:, idx] = torch.sum(pred_mip*img_fake[:, self.num_slice-idx-1:], dim=1)

        '''cacu log'''
        mse_ = self.mse(img_fake, target)
        psnr = 10 * torch.log(1 ** 2 / mse_) / np.log(10)
        ssim_ = ssim(img_fake, target, data_range=1)
        snr = cal_snr(img_fake, target)

        loss_lissim = (self.L1(img_fake, target) + 1 - ssim_) * self.lamda
        loss_mip_c1 = (self.L1(pred_mips_c1, target_mips_c1) + 1 - ssim(pred_mips_c1, target_mips_c1, data_range=1)) * self.lamda
        loss_mip_c2 = (self.L1(pred_mips_c2, target_mips_c2) + 1 - ssim(pred_mips_c2, target_mips_c2, data_range=1)) * self.lamda
        loss = loss_lissim + loss_mip_c1 + loss_mip_c2

        return {'loss': loss, 'MIP_LOSS': loss.item(), 'PSNR': psnr.item(), 
                'SSIM': ssim_.item(), 'SNR': snr}
        

class MIXloss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.lamda = options.loss.lamda
        self.num_slice = options.use_slice
        self.mse = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()

    def forward(self, img_fake, batch):
        target = batch['3D-TOF-MRA']

        '''cacu log'''
        mse_ = self.mse(img_fake, target)
        psnr = 10 * torch.log(1 ** 2 / mse_) / np.log(10)
        ssim_ = ssim(img_fake, target, data_range=1)
        snr = cal_snr(img_fake, target)

        loss_lissim = (self.L1(img_fake, target) + 1 - ssim_) * self.lamda
        loss = loss_lissim

        return {'loss': loss, 'MIP_LOSS': loss.item(), 'PSNR': psnr.item(), 
                'SSIM': ssim_.item(), 'SNR': snr}

