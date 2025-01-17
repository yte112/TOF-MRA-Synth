import os
import numpy as np
from utils.dataset import dict_as_namespace
import pydicom
from pydicom import dcmread
import json
import torch
import h5py
from model.pix2pix import GeneratorUNet
import SimpleITK as sitk
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
from pydicom.uid import ExplicitVRLittleEndian


def pd_toExcel(data, fileName):  # pd2excel
    with pd.ExcelWriter(fileName, engine='openpyxl') as writer:
        df = pd.DataFrame(data)
        df.to_excel(writer, index=False)


def createMIP(np_img, dim):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    shape = np_img.shape
    np_mip = np.zeros(shape)
    
    if dim == 0:
       for i in range(shape[0]):
           np_mip[i,:,:] = np.amax(np_img[0:i+1], 0)
    if dim == 1:
       for i in range(shape[1]):
           np_mip[:,i,:] = np.amax(np_img[:,0:i+1], 1)
    if dim == 2:
       for i in range(shape[2]):
           np_mip[:,:,i] = np.amax(np_img[:,:,0:i+1], 2)
    return np_mip


def hdf52data(root, name, modal):
    tempo_dict = {}
    with h5py.File(f'{root}{name}/{modal}.hdf5', 'r') as f:
        data = f['data'][:]
        tempo_dict['data'] = torch.tensor(data).float()
        tempo_dict['max'] = f['max'][()]
    return tempo_dict


def get_img(root, name):
    root = f'{root}/{name}/3D-TOF-MRA'
    reader = sitk.ImageSeriesReader()  # 读取3D array
    seriesIDs = reader.GetGDCMSeriesIDs(root)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(root, seriesIDs[i])
        lens[i] = len(dicom_names)
    dicom_names = reader.GetGDCMSeriesFileNames(root)

    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(root, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    return img


def save_dicom_nii(name, out, gt, dcm_root):
    if not os.path.exists(f'./inference/{name}/dicom'):
        os.makedirs(f'./inference/{name}/dicom')
    dicom_files = out['dcm_files']
    img = get_img(dcm_root, name)
    data_nii = out['synthed'] * gt['max']  # l, 290, 320
    data_nii = data_nii.astype(np.int16)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    
    # save nii
    data = np.transpose(data_nii, (0, 2, 1))  # for nii: l, 320, 290
    res = sitk.GetImageFromArray(data)
    res.SetSpacing(spacing)
    res.SetDirection(direction)
    res.SetOrigin(origin)
    sitk.WriteImage(res, f'./inference/{name}/3D-TOF-MRA.nii.gz')

    # save mip_nii
    for i in range(3):
       mip_img = createMIP(data, i)
       mip_img = sitk.GetImageFromArray(mip_img)
       mip_img.SetSpacing(spacing)
       mip_img.SetDirection(direction)
       mip_img.SetOrigin(origin)
       sitk.WriteImage(mip_img, f'./inference/{name}/MIP{i}.nii.gz')
    
    # save dicom
    series_uid = pydicom.uid.generate_uid()
    data_dcm = np.transpose(data_nii, (2, 1, 0))  # for dcm: 320, 290, l  
    uid_pool = set()
    uid_pool.add(series_uid)
    for i in range(len(dicom_files)):

        sop_uid = pydicom.uid.generate_uid()
        while sop_uid in uid_pool:
            sop_uid = pydicom.uid.generate_uid()
        uid_pool.add(sop_uid)

        src_dcm = dicom_files[i]
        TransferSyntaxUID = getattr(src_dcm.file_meta, 'TransferSyntaxUID', None)
        if TransferSyntaxUID:
            pass
        else:
            src_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        src_dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian  
        src_dcm.SeriesDescription = 'TOF_synth'
        src_dcm.SeriesInstanceUID = series_uid
        src_dcm.SOPInstanceUID = sop_uid
        src_dcm.DerivationDescription = ExplicitVRLittleEndian
        src_dcm.PixelData = data_dcm[:, :, i].tobytes()
        src_dcm.save_as(f'./inference/{name}/dicom/ImageFileName{i:03d}.dcm', write_like_original=True)  # save dicom

class Infer:
    def __init__(self, options):
        self.dcm_root = options.dcm_root
        self.hdf5_root = options.hdf5_root
        self.cuda_num = options.cuda_num
        self.num_slice = options.num_slice
        self.use_modality = options.use_modality
    
        with open("./3data.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.names = load_dict['valid'] + load_dict['test']

        net = GeneratorUNet(self.num_slice, self.num_slice)
        net.eval()
        ckpt = torch.load(f'./logs/checkpoints/epoch_{options.last_num}.ckpt', map_location=f'cuda:{self.cuda_num}')
        z = dict()
        for k, v in ckpt['state_dict'].items():
            if k[0] != 'D':
                k = k[2:]
                z[k] = v
        net.load_state_dict(z)
        self.net = net.cuda(self.cuda_num)

    def read(self, name):
        data = {}
        for modal in self.use_modality:
            data[modal] = hdf52data(self.hdf5_root, name, modal)
            data[modal]['data'] = data[modal]['data'] / data[modal]['max']

        dicom_name = os.listdir(f'{self.dcm_root}{name}/3D-TOF-MRA/')  # read dcm name
        dicom_names = [na for na in dicom_name if na[0] != '.']
        dicom_names.sort()
    
        data['dcm_files'] = [dcmread(f'{self.dcm_root}/{name}/3D-TOF-MRA/' + dn) for dn in dicom_names]  # read dcm
        return data

    def use_net(self, name):
        # use with self.read
        data = self.read(name)
        l = data[self.use_modality[0]]['data'].shape[0]
        for modal in self.use_modality:
            data[modal]['data'] = data[modal]['data'].cuda(self.cuda_num)
        data_nii = np.zeros([l, 290, 320])

        for i in range(l-self.num_slice+1):
            input = []
            for modal in self.use_modality:
                input.append(data[modal]['data'][i:i+self.num_slice].unsqueeze(0))

            out = self.net(torch.cat(input, 1))
            # out = torch.log(out)
            out = out.squeeze(0)
            out = out.data.cpu().numpy()
        
            data_nii[i:i+self.num_slice, :288] += out

        for i in range(self.num_slice - 1):
            data_nii[i] = data_nii[i] / (i + 1)
            data_nii[l-i-1] = data_nii[l-i-1] / (i+1)
        for i in range(self.num_slice - 1, l - self.num_slice + 1):
            data_nii[i] = data_nii[i] / self.num_slice

        out = {}
        out['synthed'] = data_nii
        out['dcm_files'] = data['dcm_files']

        return out
    
    def save_cacu(self, name, gt):
        out = self.use_net(name)
        save_dicom_nii(name, out, gt, self.dcm_root)

        denoised = out['synthed']
        gt_array = gt['data'] / gt['max']
        target = np.zeros_like(denoised)
        target[:, :288] = gt_array

        max_val = target.max()
        mse = np.sum(np.abs(target-denoised)**2)/target.size
        psnr = 10*np.log10(max_val ** 2 / mse)
        ssim = compare_ssim(denoised, target, data_range=max_val)
        return psnr, ssim
    
    def infer_all(self):
        result = {'case': [], 'psnr': [], 'ssim': []}
        for name in self.names:
            print(f'start infer {name}')
            gt = hdf52data(self.hdf5_root, name, '3D-TOF-MRA')
            psnr, ssim = self.save_cacu(name, gt)
            print(f'psnr: {psnr}, ssim: {ssim}\n')
            result['case'].append(name)
            result['psnr'].append(psnr)
            result['ssim'].append(ssim)
        pd_toExcel(result, f'infer.xlsx')


if __name__ == '__main__':
    options = {
        'cuda_num': 4,
        'last_num': None,  # pth num
        'dcm_root': 'your dicom root',
        'hdf5_root': 'your hdf5 root',
        'use_modality': ['3D-T1W', '3D-T2W', '3D-FLAIR'],
        'num_slice': 5
    }
    options = dict_as_namespace(options)
    infer_pet_t2 = Infer(options)
    infer_pet_t2.infer_all()


                
    