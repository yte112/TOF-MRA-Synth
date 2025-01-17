from tqdm import tqdm
import random
import numpy as np
import h5py
import json
import torch
from torch.utils.data import Dataset
from types import SimpleNamespace


def dict_as_namespace(d) -> SimpleNamespace:
    """
    Convert a dictionaty to a namespace (i.e., support for the `.` notation)
    """
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_as_namespace(v))
        else:
            setattr(x, k, v)
    return x
    

def gauss_noise_matrix(matrix, sigma):
    mu = 0
    shape = matrix.shape
    noise_matrix = np.random.normal(mu, sigma, size=shape).astype(np.float32)
    noise_matrix += matrix
    return noise_matrix


def rand_add(slice, t_v, model):
    if t_v == 'val' or model == '3D-TOF-MRA':
        return slice
    ra = random.random()
    if ra < 0.1:
        return gauss_noise_matrix(slice, 0.1)
    if ra < 0.3:
        return gauss_noise_matrix(slice, 0.05)
    if ra < 0.5:
        return gauss_noise_matrix(slice, 0.03)
    else:
        return slice
    

def add_minus(batch):
    # slow
    use_modality = batch[0].keys()
    ra = random.random()
    l = len(batch)
    if ra < 0.4:
        for i in range(l//2):
            for modal in use_modality:
                batch[i][modal], batch[l-i-1][modal] = \
                batch[i][modal] + batch[l-i-1][modal], batch[l-i-1][modal] + batch[i][modal]
    new_batch = {}
    for modal in use_modality:
        new_batch[modal] = torch.cat([torch.unsqueeze(batch[i][modal], 0) for i in range(l)], 0)
    return new_batch
    


def hdf52data(root, name, modal, core, num_slice):
    num = num_slice // 2
    with h5py.File(f'{root}/{name}/{modal}.hdf5', 'r') as f:
        slices = f['data'][:][core-num:core+num+1]
        max = f['max'][()]
        slices = slices / max
    return slices


def read_data(root, name, modal):
    with h5py.File(f'{root}/{name}/{modal}.hdf5', 'r') as f:
        slices = f['data'][:]
        max_len = f['max_len'][()]
        max = f['max'][()]
        '''
        mean = f['mean'][()]
        std = f['std'][()]
        slices = (slices - mean) / std
        if modal == '3D-TOF-MRA':
            slices = np.exp(slices)
        '''
        slices = slices/max
        
        
        # slices = 100/(slices+1)
    return slices, max_len


def get_max_len(root, name):
    with h5py.File(f'{root}/{name}/3D-TOF-MRA.hdf5', 'r') as f:
        return f['max_len'][()]
    

class MriForGANSingle(Dataset):
    def __init__(self, train_val, num_slice, root, use_modality):
        self.root = root
        self.use_modelity = use_modality + ['3D-TOF-MRA']
        self.num_slice = num_slice
        self.t_v = train_val

        with open("./3data.json",'r') as load_f:
            load_dict = json.load(load_f)

        self.names = load_dict[train_val]

        self.max_len = {}
        for i in tqdm(range(len(self.names)), desc=f'read data len', total=len(self.names)):
            self.max_len[self.names[i]] = get_max_len(root, self.names[i])

        self.use_data = []
        for name in self.names:
            for core in range(num_slice//2, self.max_len[name]-num_slice):
                self.use_data.append([name, core])
        random.shuffle(self.use_data)

    def __getitem__(self, idx):
        batch = {}
        num = self.num_slice // 2
        name, core = self.use_data[idx]
        for modal in self.use_modelity:
            data_model, _ = read_data(self.root, name, modal)
            slice = data_model[core-num:core+num+1]
            slices = torch.tensor(rand_add(slice, self.t_v, modal))
            # slices = torch.tensor(slice)
            batch[modal] = slices
        # batch['source'] = torch.cat([data[i] for i in ['3D-T1W', '3D-T2W', '3D-FLAIR']], 0)
        # batch['source'] = data['3D-T1W']
        # batch['target'] = data['3D-TOF-MRA']
       
        if random.random() < 0.5:
            for modal in self.use_modelity:
                batch[modal] = torch.flip(batch[modal], [2])
       
        return batch
    
    
    def __len__(self):
        return len(self.use_data)
    

class MriForGAN(Dataset):
    def __init__(self, train_val, num_slice, root, use_modality):
        self.use_modelity = use_modality + ['3D-TOF-MRA']
        self.num_slice = num_slice
        self.t_v = train_val
        
        with open("./3data.json",'r') as load_f:
            load_dict = json.load(load_f)

        self.names = load_dict[train_val]

        self.data = {}
        self.max_len = {}
        for i in range(len(self.names)):
            print(f'load data {i}/{len(self.names)}')
            self.data[self.names[i]] = {}
            for modal in self.use_modelity:
                self.data[self.names[i]][modal], self.max_len[self.names[i]] = \
                read_data(root, self.names[i], modal)

        self.use_data = []
        for name in self.names:
            for core in range(num_slice//2, self.max_len[name]-num_slice):
                self.use_data.append([name, core])
        random.shuffle(self.use_data)

        '''
        if train_val == 'train':
            self.mix_up(40)
        '''

    def __getitem__(self, idx):
        batch = {}
        num = self.num_slice // 2
        name, core = self.use_data[idx]
        for modal in self.use_modelity:
            slice = self.data[name][modal][core-num:core+num+1]
            slices = torch.tensor(rand_add(slice, self.t_v, modal))
            batch[modal] = slices

        if random.random() < 0.5:
            for modal in self.use_modelity:
                batch[modal] = torch.flip(batch[modal], [2])
        return batch
    
    
    def __len__(self):
        return len(self.use_data)
    
    
    def mix_up(self, mix_num):
        mix_name = self.names[:mix_num]
        for i in range(mix_num//2):
            name1 = mix_name[i]
            name2 = mix_name[i+1]
            l = min(self.max_len[name1], self.max_len[name2])
            for modal in self.use_modelity:
                self.data[name1][modal][:l] = (self.data[name1][modal][:l] + self.data[name2][modal][:l]) / 2
                '''
                self.data[name1][modal][:l], self.data[name2][modal][:l] = \
                self.data[name1][modal][:l] - self.data[name2][modal][:l], self.data[name1][modal][:l] + self.data[name2][modal][:l] 
                '''

                

                





