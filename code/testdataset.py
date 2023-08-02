import torch
import os
import numpy as np
from scipy import io
import random
from torch.utils.data import DataLoader, Dataset
#from torch.utils.data import _utils
import torch
path='./SV/layernum/test/'
#path='./SV/UPA/UPA_test/'
#path='./Deep/testdataset_ULA/'

def test_data(i):
    a=i
    #dbnum=(-10,-5,0,5,10,15,20)
    dbnum=(-10,-5,0,5,10,15,20)
    n=dbnum[a]
    print(n)
    val_samples=io.loadmat(os.path.join(path + 'SV_UPA_test_128_measurements_%ddB.mat')%(n))['y']
    y=val_samples.T
    yy=[]
    for j in range(len(y)):
        r=y[j].real
        i=y[j].imag
        yy.append(r)
        yy.append(i)
    y=yy
    y=np.array(y)
    y=torch.tensor(y)
    y=torch.reshape(y,[-1,2,128])
    val_labels=io.loadmat(os.path.join(path + 'SV_UPA_test_channel.mat'))['x']
    h=val_labels.T
    hh=[]
    for j in range(len(h)):
        r=h[j].real
        i=h[j].imag
        hh.append(r)
        hh.append(i)
    h=hh
    h=np.array(h)
    h=torch.tensor(h)
    h=h.float()
    h=torch.reshape(h,[-1,2,16,16])
    return y,h

class TensorDataset(Dataset):


    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


