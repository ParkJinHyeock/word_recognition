import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

torch.manual_seed(215)
torch.cuda.manual_seed(215)
torch.cuda.manual_seed_all(215)

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(213)
np.random.seed(214)
torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.nn.functional as F
import math


class FC_Net(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=64, batch_size=32):
        super().__init__()
        self.n_channel = n_channel
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=[3, 3], stride=[1, 1])
        self.pool1 = nn.MaxPool2d((1,2))
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=[3, 3], stride=[1, 1])
        self.pool2 = nn.MaxPool2d((1,2))
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=[3, 3], stride=[1, 1])
        self.pool3 = nn.MaxPool2d((2,2))
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=[3, 3], stride=[1, 1])
        self.bn3 = nn.BatchNorm2d(n_channel*2)
        self.gru1 = nn.GRU(input_size=65, hidden_size=100, num_layers=2, bidirectional=True)
        self.output = n_output
        self.weight = None
        self.flat = nn.Flatten()
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        self.batch = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_1 = torch.nn.Linear(n_output, 3*n_output)
        self.linear_2 = torch.nn.Linear(3*n_output, n_output)

    def forward(self, x):
        # x = x.permute(0, 1, 3, 2)
        # x = torch.reshape(x, [x.shape[0], x.shape[2], x.shape[3]])
        # x, _ = self.gru1(x)
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])
        # x = x.permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.drop1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 1, 3, 2)
        x = self.flat(x)
        if self.weight is None:
           self.weight = nn.Parameter(torch.randn(self.output, x.size()[1])).to(self.device)
        x = F.linear(x, self.weight)
        # x = F.relu(x)
        # x = self.linear_1(x)
        # x = F.relu(x)
        # x = self.linear_2(x)
        # x = self.drop3(x)
        return F.log_softmax(x, dim=1)



class sample_NET(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=64, batch_size=32):
        super().__init__()
        self.n_channel = 128
        self.output = n_output

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm1d(n_channel)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(3)

        self.conv3 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(2*n_channel)
        self.pool3 = nn.MaxPool1d(3)

        self.conv4 = nn.Conv1d(2*n_channel, 2*n_channel, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(2*n_channel)
        self.pool4 = nn.MaxPool1d(3)

        self.conv5 = nn.Conv1d(2*n_channel, 2*n_channel, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm1d(2*n_channel)
        self.pool5 = nn.MaxPool1d(3)

        self.conv6 = nn.Conv1d(2*n_channel, 2*n_channel, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm1d(2*n_channel)
        self.pool6 = nn.MaxPool1d(3)

        self.weight = None
        self.flat = nn.Flatten()
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)
        self.batch = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_1 = torch.nn.Linear(n_output, 3*n_output)
        self.linear_2 = torch.nn.Linear(3*n_output, n_output)
        self.linear_3 = torch.nn.LazyLinear(self.output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)


        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.drop2(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool5(x)

        x = self.flat(x)

        # if self.weight is None:
        #    self.weight = nn.Parameter(torch.randn(self.output, x.size()[1])).to(self.device)
        # x = F.linear(x, self.weight)
        x = self.linear_3(x)

        return F.log_softmax(x, dim=1)


    
class SpinalVGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=10):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        Half_width =128
        layer_width =128
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.2), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.2), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.2), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.2), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.2), nn.Linear(layer_width*4, num_classes),)
        
    
    def forward(self, x):
        Half_width =128
        layer_width =128
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,Half_width:2*Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)

        return F.log_softmax(x, dim=1)

######################################################################################################


import torch
from torch import Tensor
import torch.nn as nn


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        # out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        # out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(torch.nn.Module):
    def __init__(self, output_size):
        super(BCResNet, self).__init__()
        self.output = output_size
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16, 8)
        self.block1_2 = BroadcastedBlock(8)

        self.block2_1 = TransitionBlock(8, 12, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12, 16, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16, 20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20, 20, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(32, 12, 1, bias=False)
        self.weight= None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):

        out = self.conv1(x)

        out = self.block1_1(out)
        out = self.block1_2(out)

        out = self.block2_1(out)
        out = self.block2_2(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        out = self.conv4(out)
        out = torch.reshape(out, [out.shape[0], -1])
        if self.weight is None:
           self.weight = nn.Parameter(torch.randn(self.output, out.size()[1])).to(self.device)
        out = F.linear(out, self.weight)

        return out

class MarbleNet(nn.Module):
  def __init__(self, num_classes, C=130):
    super(MarbleNet, self).__init__()
    dropout = 0.2
    self.prologue = nn.Sequential(
      nn.Conv1d(65, C, groups=65, kernel_size=11, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU(inplace=True)
    )

    self.sub00 = nn.Sequential(
      nn.Conv1d(C, C, kernel_size=13, groups=C, padding='same', bias=False),
      nn.Conv1d(C, C//2, kernel_size=1,padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Conv1d(C//2, C//2, kernel_size=13, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
    )

    self.sub01 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=13, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub02 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub0C = nn.Sequential(
      nn.Conv1d(C, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub10 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=15, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Conv1d(C//2, C//2, kernel_size=15, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
    )

    self.sub11 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=15, groups=C//2, padding='same', bias=False), 
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub12 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub1C = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub20 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=17, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Conv1d(C//2, C//2, kernel_size=17, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
    )

    self.sub21 = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=17, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.sub22 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub2C = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.epi1 = nn.Sequential(
      nn.Conv1d(C//2, C, groups=C//2, kernel_size=29, dilation=2, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU()
    )

    self.epi2 = nn.Sequential(
      nn.Conv1d(C, C, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU()
    )

    self.epi3 = nn.Conv1d(C, num_classes, kernel_size=1, bias=True)
    self.sigmoid = nn.LogSoftmax(dim=-2)

  def forward(self, input):
    input = torch.squeeze(input)
    if len(input.size()) == 2:
      input = torch.unsqueeze(input, 0)
    x = self.prologue(input)
    x_ = self.sub0C(x)
    x = self.sub00(x)
    x = self.sub01(x)
    
    x = x + x_
    x = self.sub02(x)

    x_ = self.sub1C(x)
    x = self.sub10(x)
    x = self.sub11(x)
    x = x + x_
    x = self.sub12(x)

    x_ = self.sub2C(x)
    x = self.sub20(x)
    x = self.sub21(x)
    x = x + x_
    x = self.sub22(x)

    x = self.epi1(x)
    # x = self.epi2(x)
    x = torch.mean(x, dim=2, keepdim=True)
    x = self.epi3(x)
    x = self.sigmoid(x)
    x = torch.squeeze(x, -1)
    return x


class CNN_TD(nn.Module):
  def __init__(self, num_classes):
    super(CNN_TD, self).__init__()
    fsize = 32
    td_dim = 256

    self.ConvMPBlock_1 = self.ConvMPBlock(num_conv=2, fsize=fsize, in_channel=1)
    self.ConvMPBlock_2 = self.ConvMPBlock(num_conv=2, fsize=2*fsize, in_channel=fsize)
    self.ConvMPBlock_3 = self.ConvMPBlock(num_conv=3, fsize=4*fsize, in_channel=fsize*2)    
    self.linear_1 = nn.Linear(768, td_dim)
    self.linear_2 = nn.Sequential(nn.Linear(td_dim, 128), nn.BatchNorm1d(128), nn.ReLU())
    self.linear_3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
    self.linear_4 = nn.Sequential(nn.Linear(64, num_classes), nn.LogSoftmax())

  def ConvMPBlock(self, num_conv=2, fsize=32,  in_channel=None, kernel_size=3, pool_size=(2,2), strides=(2,2), BN=True, DO=True, MP=True):
      mod_list = []
      for i in range(num_conv):
        if i == 0:
          mod_list.append(nn.Conv2d(in_channel, fsize, kernel_size, padding='same'))
        else:
          mod_list.append(nn.Conv2d(fsize, fsize, kernel_size, padding='same'))
        if BN:
          mod_list.append(nn.BatchNorm2d(fsize))
        if DO:
          mod_list.append(nn.Dropout(0.1))
        mod_list.append(nn.ReLU())
      if MP:
        mod_list.append(nn.MaxPool2d(kernel_size=pool_size, stride=strides))
      return nn.Sequential(*mod_list)  

  def forward(self, input):
    # input channel * time * freq
    x = self.ConvMPBlock_1(input)
    x = self.ConvMPBlock_2(x)
    x = self.ConvMPBlock_3(x)
    x = torch.transpose(x, 1, 2)
    x = torch.reshape(x, (x.size(0), x.size(1), x.size(2)*x.size(3)))
    x = self.linear_1(x)
    x = torch.mean(x, dim=1)
    x = self.linear_2(x)
    x = self.linear_3(x)
    x = self.linear_4(x)    
    return x