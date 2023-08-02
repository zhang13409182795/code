from torch.autograd import Variable
from torch.nn import init

from dataset import *
import torch.nn as nn
import torch
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


# Define  Stage
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 34, 3, 3)))
        self.RB1 = ResidualBlock(32, 32, 3, bias=True)
        self.RB2 = ResidualBlock(32, 32, 3, bias=True)
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        #self.ConvLSTM = ConvLSTM(32, 32, 3)

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



# Define Network
class Network(torch.nn.Module):
    def __init__(self, LayerNo):
        super(Network, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        channels=32
        self.channels = channels

        A = io.loadmat(os.path.join(path, 'CSmatrix256128.mat'))['A']
        self.Phi = nn.Parameter(torch.from_numpy(A).float(), requires_grad=True)#128*256
        self.PhiT = nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True)#256*128
        self.lambda_step = nn.Parameter(torch.Tensor([1]), requires_grad=True)

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(2, channels, 3, padding=1, bias=True)

    def forward(self, y):

        y = torch.reshape(y, [-1, 128])#32*128
        y = torch.tensor(y, dtype=torch.float32).cuda()
        y = torch.transpose(y, -1, 0)#128*32
        y = y.float()
        x1 = torch.matmul(self.PhiT, y).cuda()  #256*32
        x1 = torch.tensor(x1, dtype=torch.float32).cuda()
        Rk1 = torch.transpose(x1, -1, 0)#32*256
        Rk2 = torch.reshape(Rk1, [-1, 2, 16, 16])#16*2*16*16
        x = Rk2

        PhiTPhi = torch.mm(self.PhiT, self.Phi)
        PhiTb = Rk1  # 64,256


        [h, c] = [None, None]
        z = (self.fe(x)).cuda()

        for i in range(self.LayerNo):
            x, z, h, c = self.fcs[i](Rk1, z, h, c,PhiTPhi,PhiTb)

        x_final = x

        return x_final


def train(model, opt, train_loader, epoch, batch_size):
    model.train()
    n = 0
    for y,h in train_loader:
        n = n + 1
        opt.zero_grad()
        y = Variable(y.float().cuda())
        h = Variable(h.float().cuda())
        outputs= model(y)
        loss = torch.mean((outputs-h)**2)
        #mse = ((h - outputs) ** 2).mean(-1).mean(-1).squeeze()
        #loss = torch.sqrt((torch.sqrt(loss) ** 2).mean())
        loss.backward()
        opt.step()
        if n % 25 == 0:
            output = "epoch: %d  %02d loss: %.4f " % (
            epoch,batch_size*n,loss.data.item())
            print(output)
    return loss


def get_val_result(model,val_loader,is_cuda=True):
    model.eval()
    with torch.no_grad():
        nmse = []
        for data, target in val_loader:
            outputs = model(data)
            outputs = outputs.cuda()
            target = target.cuda()
            # loss = torch.mean((outputs-target)**2)
            nmse_db = 10 * torch.log10(torch.mean((abs(outputs - target)) ** 2) / torch.mean((abs(target)) ** 2))
            nmse.append(nmse_db)
        # out=loss.numpy()
        # out = np.mean(loss)
        out = sum(nmse) / len(nmse)
    return out


if __name__ == "__main__":
    is_cuda=True
    learning_rate = 0.0001
    EpochNum = 300
    batch_size =16
    PhaseNum=1

    results_saving_path = "results"

    net_name = "SV_UPA_layernum"

    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    results_saving_path = os.path.join(results_saving_path, net_name)
    if not os.path.exists(results_saving_path):
            os.mkdir(results_saving_path)

    print('Load Data...')
    train_dataset = tensor_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    val_dataset = tensor_dataset_val
    val_loader = DataLoader(tensor_dataset_val, batch_size=batch_size,
                        shuffle=True, num_workers=2)
    print('Load completedly!')

    model = Network(PhaseNum)
    #model.apply(weights_init)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    model.to(device)
    #model.cuda()
    scheduler1 = StepLR(opt, step_size=100, gamma=0.1)
    sub_path = os.path.join(results_saving_path,str(PhaseNum))
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
    best_mse_val = 10000000000000
    best_mse_test = 10000000000000
    for epoch in range(1, EpochNum + 1):
        one_mse_test = train(model, opt, train_loader, epoch, batch_size)
        print("epoch:%d学习率：%f" % (epoch, opt.param_groups[0]['lr']))
        scheduler1.step()
        one_mse_val = get_val_result(model, val_loader, is_cuda=True)
        print_str = " epoch: %d  nmse_db: %.4f" % (epoch, one_mse_val)
        print(print_str)
        if one_mse_test < best_mse_test:
            best_mse_test = one_mse_test
            torch.save(model.state_dict(), sub_path + "/best_model_test.pkl")
        if one_mse_val < best_mse_val:
            best_mse_val = one_mse_val
            # best_model=copy.deepcopy(model)
            # model=best_model
            torch.save(model.state_dict(), sub_path + "/best_model_val.pkl")
            # torch.save(model.state_dict(), "./best_model.pkl")









