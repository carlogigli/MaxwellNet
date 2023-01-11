import torch
import numpy as np
import os


class LensDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, filename):
        dataset = np.load(os.path.join(data_path, filename + ".npz"))
        self.ref_index = dataset['n']
        self.power = dataset['S0']

    def __len__(self):
        return self.ref_index.shape[0]

    def __getitem__(self, idx):
        sample = self.ref_index[idx, :, :, :]
        power = self.power[idx]
        return sample, power, idx

class RI2D_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, filename):
        dataset = np.load(os.path.join(data_path, filename + ".npz"))
        self.ref_index = dataset['n']
        self.power = dataset['S0']

    def __len__(self):
        return self.ref_index.shape[0]

    def __getitem__(self, idx):
        sample = self.ref_index[idx, :, :, :]
        power = self.power[idx]
        return sample, power, idx

class WG2D_Dataset(torch.utils.data.Dataset):
    """
    2D Waveguide with modal excitation. It is the same dataset as RI2D_Dataset but it includes also
    the excitation current distibution
    """
    def __init__(self, data_path, filename):
        dataset = np.load(os.path.join(data_path, filename + ".npz"))
        self.ref_index = dataset['n']
        self.power = dataset['S0']
        self.loc_src = dataset['src']
        self.chi3 = dataset['chi3']

    def __len__(self):
        return self.ref_index.shape[0]

    def __getitem__(self, idx):
        sample = self.ref_index[idx, :, :, :]
        power = self.power[idx]
        return sample, power, idx