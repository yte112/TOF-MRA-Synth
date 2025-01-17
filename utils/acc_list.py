import os
import sys
import torch
import pandas as pd
import PIL.Image as Image
from torchvision.utils import make_grid


def excel_dict(root, keys, mode, save_data=None):
    if mode == 'read':
        data = pd.read_excel(root)
        excel_dict = dict(zip(keys, [list(data[key]) for key in keys]))
        return excel_dict
    else:
        df = pd.DataFrame(save_data)  # 创建DataFrame
        df.to_excel(root, index=False)  # 


def sample_images(epoch, batch, G, use_modality):
    """Saves a generated sample from the validation set"""
    input = torch.cat([batch[modal] for modal in use_modality], 1)
    fake_G = G(input)
    real = batch['3D-TOF-MRA'][:, 0:1].data 
    fake = fake_G[:, 0:1].data 
    # real, fake = torch.log(real), torch.log(fake)
    real = (real - real.min()) / (real.max() - real.min())
    fake = (fake - fake.min()) / (fake.max() - fake.min())
    img = torch.cat((real, fake), -2)
    grid = make_grid(img)
    ndarr = grid.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.convert('L').save(f"logs/images/{epoch + 1}.png")


class TestImage:
    def __init__(self, start):
        self.keys = ['MIP_LOSS', 'PSNR', 'SSIM', 'SNR']
        self.train_log = {'MIP_LOSS': [], 'PSNR': [], 'SSIM': [], 'SNR': []}
        self.val_log = {'MIP_LOSS': [], 'PSNR': [], 'SSIM': [], 'SNR': []}
        if start != 0:
            for key in self.keys:
                    self.train_log[key] = excel_dict('./logs/train.xlsx', self.keys, 'read')[key][0:start]
                    self.val_log[key] = excel_dict('./logs/val.xlsx', self.keys, 'read')[key][0:start]

    def make_list(self, loss, batch_idx, len_, t_v):  # cacu every batch
        if batch_idx == 0 and t_v == 'train':
            for key in self.keys:
                self.train_log[key].append(0)
        if batch_idx == 0 and t_v == 'val':
            for key in self.keys:
                self.val_log[key].append(0)

        if t_v == 'train':
            for key in self.keys:
                self.train_log[key][-1] += loss[key] / len_

        if t_v == 'val':
            for key in self.keys:
                self.val_log[key][-1] += loss[key] / len_

    def save_results(self):  # save every epoch
        excel_dict('./logs/train.xlsx', self.keys, 'write', self.train_log)
        excel_dict('./logs/val.xlsx', self.keys, 'write', self.val_log)

    def show(self, t_v):  # print every epoch
        if t_v == 'train':
            print(
                "\r[MIP_LOSS: %F, PSNR: %f, SSIM: %f, SNR: %f]"
                % (
                    self.train_log['MIP_LOSS'][-1],
                    self.train_log['PSNR'][-1],
                    self.train_log['SSIM'][-1],
                    self.train_log['SNR'][-1]
                )
            )
        else:
            print(
                "\r[MIP_LOSS: %F, PSNR: %f, SSIM: %f, SNR: %f]"
                % (
                    self.val_log['MIP_LOSS'][-1],
                    self.val_log['PSNR'][-1],
                    self.val_log['SSIM'][-1],
                    self.val_log['SNR'][-1]
                )
            )
            
    def del_more_val(self):
        for key in self.keys:
                self.val_log[key] = self.val_log[key][:-1]
