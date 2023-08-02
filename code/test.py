import matplotlib.pyplot as plt
import testdataset
import os
import numpy as np
import glob
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from scipy.io import savemat
import scipy.io as sio
from torch.nn import init
from dataset import *
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define RB
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x


# Define Stage
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 34, 3, 3)))
        self.RB1 = ResidualBlock(32, 32, 3, bias=True)
        self.RB2 = ResidualBlock(32, 32, 3, bias=True)
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        

    def forward(self, Rk1,z, h, c,PhiTPhi,PhiTb):

        x = Rk1 - self.lambda_step * torch.mm(Rk1, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x = x.view(-1, 2, 16, 16)
        x_input = x
        x_a = torch.cat([x_input, z], 1)
        x_D = F.conv2d(x_a, self.conv_D, padding=1)
        x = self.RB1(x_D)
        x_backward = self.RB2(x)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + x_G

        return x_pred, x_backward, h, c

class Network(torch.nn.Module):
    def __init__(self, LayerNo,A):
        super(Network, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        channels=32
        self.channels = channels

        self.Phi = nn.Parameter(torch.from_numpy(A).float(), requires_grad=True)#128*256
        self.PhiT = nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True)#256*128
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(2, channels, 3, padding=1, bias=True)

    def forward(self, y):

        y = torch.reshape(y, [-1, 128])#32*128
        y = torch.tensor(y, dtype=torch.float32)
        y = torch.transpose(y, -1, 0)#128*32
        y = y.float()
        x1 = torch.matmul(self.PhiT, y) #256*32
        x1 = torch.tensor(x1, dtype=torch.float32)
        Rk1 = torch.transpose(x1, -1, 0)#32*256
        Rk2 = torch.reshape(Rk1, [-1, 2, 16, 16])#16*2*16*16
        x = Rk2

        PhiTPhi = torch.mm(self.PhiT, self.Phi)
        PhiTb = Rk1  # 64,256


        [h, c] = [None, None]
        z = (self.fe(x))

        for i in range(self.LayerNo):
            x, z, h, c = self.fcs[i](Rk1, z, h, c,PhiTPhi,PhiTb)

        x_final = x

        return x_final


def nmse_loss(outputs, targets):
    outputs=outputs
    h_test=[]
    for e in range(len(outputs)):
        real=np.array(outputs[e,0,:,:])
        imag=np.array(outputs[e,1,:,:])
        hir=real+imag*1j
        h=np.reshape(hir,(256))
        h_test.append(h)
    outputs=np.array(h_test)
    outputs=np.reshape(outputs,(-1,256))
    targets=targets
    h_real=[]
    for e in range(len(targets)):
        real=np.array(targets[e,0,:,:])
        imag=np.array(targets[e,1,:,:])
        hir=real+imag*1j
        y=np.reshape(hir,(256))
        h_real.append(y)
    targets=np.array(h_real)
    targets=np.reshape(targets,(-1,256))
    return outputs,targets


def get_test_result(model,test_loader):
    model.eval()
    with torch.no_grad():
        model.eval()
        hpre=[]
        hreal=[]
        for data,target in test_loader:
            data=data
            target=target
            outputs = model(data)
            xhat,x=nmse_loss(outputs,target)
            hpre.append(xhat)
            hreal.append(x)

    return hpre,hreal

def load_sampling_matrix():
    path = "./"
    data = io.loadmat(os.path.join(path, 'CSmatrix256128.mat'))['A']
    return data

if __name__ == "__main__":
    #is_cuda = True

    PhaseNum =8           
    results_saving_path = "results"

    net_name = "SV_UPA_layernum"

    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    results_saving_path = os.path.join(results_saving_path, net_name)
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)
    nmseDb=[]
    for i in range(0,7,1):
        data_tensor_test ,target_tensor_test=testdataset.test_data(i)
        tensor_dataset_test=testdataset.TensorDataset(data_tensor_test,target_tensor_test)
        test_dataset=tensor_dataset_test
        test_loader = DataLoader(test_dataset, batch_size=1,
                                                   shuffle=False, num_workers=2)
        #model.cuda()
        sub_path = os.path.join(results_saving_path,str(PhaseNum))
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        path=os.path.join(sub_path + "/best_model_test.pkl")
        A = load_sampling_matrix()
        model = MADUN(PhaseNum,A)
        model.load_state_dict(torch.load(path,map_location='cuda'))
        hpre,hreal=get_test_result(model,test_loader)
        hpre=np.reshape(hpre,(2000,256))
        hreal=np.reshape(hreal,(2000,256))
        #hreal=torch.tensor(hreal)
        #hpre=torch.tensor(hpre)
        nmse_db= 10*np.log10(np.mean((abs(hpre-hreal))**2)/np.mean((abs(hreal))**2))
        print(nmse_db)
        nmseDb.append(nmse_db)


    snr=[-10,-5,0,5,10,15,20]
    plt.plot(snr,nmseDb,c='black',marker='o', linestyle='-')
    plt.xlim(-10, 20, 5)
    plt.show()
    nmse_matlab = np.array(nmseDb)
    sio.savemat('val1020dB_UPA.mat', {'val1020dB_UPA':nmse_matlab})
    # p=dict(hpre=hpre)
    # r=dict(hreal=hreal)
    # savemat(str(sub_path)+'./hpre.mat', p)
    # savemat(str(sub_path)+'./hreal.mat', r)
    






































# net = AMP_net_Deblock()
# # 创建网络对象net
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# 选出优化器，learning rate设置为0.01， 动量设置为0.9
# 优化出合适的[w1, b1, w2, b2, w3, b3]
# train_loss = []
# for epoch in range(3):
# # 每次求导都需要更新，因此进行迭代，这里进行三层迭代
#     for batch_idx, (x, y) in enumerate(train_loader):
#         # 每次迭代会对整个数据循环一遍
#         # 这里得到的x的shape为[512, 1, 28, 28]
#         # x:[b, 1, 28, 28], y:[512]
#         # 实际图片为4维，这里需要将其转化为2维。即将x进行打平， 将[b, 1, 28, 28] ==> [b, feature]
#         x = x.view(x.size(0), 28*28)
#         out = net(x)
#         # 将图片传给x
#         y_onehot = one_hot(y)
#         # 对y进行one_hot编码
#         loss = F.mse_loss(out, y_onehot)
#         optimizer.zero_grad()
#         # 清零操作
#         # 采用均方差计算loss
#         loss.backward()
#         optimizer.step()
#         # 完成w' = w - lr*gradient的过程，更新梯度
#     # 迭代三次，训练过程结束后，得到最佳的[w1, b1, w2, b2, w3, b3]
#         train_loss.append(loss.item())
#         if batch_idx % 10 == 0:
#             print(epoch, batch_idx, loss, loss.item())
#              # 每隔10个输出结果
#
#
#
