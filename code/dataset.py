import torch
import os
import numpy as np
from scipy import io
import random
from torch.utils.data import DataLoader, Dataset
import torch
#path='./SV/UPA/UPA_training/'
path='./SV/layernum/training/'
#path='./Deep/ULA_train/'
def train_data():
    train_samples=io.loadmat(os.path.join(path+'SV_UPA_training_128_measurements_15dB.mat'))['y']
    y=train_samples.T
    y_=[]
    for j in range(len(y)):
        r=y[j].real
        i=y[j].imag
        y_.append(r)
        y_.append(i)
    y=y_
    # y = y.float()
    y=np.array(y)
    y=torch.tensor(y)
    y=torch.reshape(y,[-1,2,128])
    # print(y.shape)
    # print(y)
    # print('y')
    train_labels=io.loadmat(os.path.join(path + 'SV_UPA_training_channel.mat'))['x']
    h=train_labels.T
    h_=[]
    for j in range(len(h)):
        r=h[j].real
        i=h[j].imag
        h_.append(r)
        h_.append(i)
    h=h_
    h=np.array(h)
    h=torch.tensor(h)
    h=h.float()
    h=torch.reshape(h,[-1,2,16,16])
    # print(h)
    # print('h')


    return y,h
def val_data():
    val_samples=io.loadmat(os.path.join(path +'SV_UPA_val_128_measurements_15dB.mat'))['y']
    y=val_samples.T
    y_=[]
    for j in range(len(y)):
        r=y[j].real
        i=y[j].imag
        y_.append(r)
        y_.append(i)
    y=y_
    # y = y.float()
    y=np.array(y)
    y=torch.tensor(y)
    y=torch.reshape(y,[-1,2,128])
    val_labels=io.loadmat(os.path.join(path + 'SV_UPA_validation_channel.mat'))['x']
    h=val_labels.T
    h_=[]
    for j in range(len(h)):
        r=h[j].real
        i=h[j].imag
        h_.append(r)
        h_.append(i)
    h=h_
    h=np.array(h)
    h=torch.tensor(h)
    h=h.float()
    h=torch.reshape(h,[-1,2,16,16])
    return y,h


y,h=train_data()
yy,hh=val_data()
#print(yy.shape)
#print(hh.shape)
#yyy,hhh=test_data()
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]

# 生成数据

data_tensor = y

# print(y.shape)
target_tensor = h
# print(h.shape)
data_tensor_val=yy
target_tensor_val=hh
# data_tensor_test=yyy
# target_tensor_test=hhh

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)
tensor_dataset_val=TensorDataset(data_tensor_val,target_tensor_val)
# tensor_dataset_test=TensorDataset(data_tensor_test,target_tensor_test)

