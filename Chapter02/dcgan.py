import torch
import numpy as np
import torchvision
from torchvision import datasets  # 导入mnist数据
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # 注意此处的 D a t a L o a d e r大小写问题
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
num_workers = 2
batch_size = 128
image_size = 28  # mnist原始图片是28*28*1
nc = 1  # number of channels
nz = 100  # size of latent vector
ngf = 64  # size of  feature maps in gener
ndf = 64  # size of feature maps in discri
num_epoches = 5
lr = 2e-4
beta1 = 0.5  # hyperpara for adam
ngpu = 1   # num of gpu
device = torch.device("cuda:0" if (torch.cuda.is_available()) and ngpu > 0 else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# Compose函数将多种transform整合到一起, 可以看出是传入一个list作为参数, Normalize后的参数分别为均值和方差, 此处单通道故为单值,三通道要三个值
train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
real_batch = next(iter(train_loader))
plt.figure(figsize=(8, 8))
print(real_batch[0].shape)
# # print(torchvision.utils.make_grid(real_batch[0][:64].shape))
# plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
#
# plt.show()


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Disciminator(nn.Module):
    super(Disciminator, self).__init__()
    def __int__(self, ngpu):
        nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


