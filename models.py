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
    def __init__(self, n_input=1, n_output=35, stride=1, n_channel=64, batch_size=32):
        super().__init__()
        self.n_channel = 16
        self.output = n_output
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm1d(n_channel)

        self.conv1_1 = nn.Conv1d(n_channel, n_channel, kernel_size=3, stride=3)
        self.bn1_1 = nn.BatchNorm1d(n_channel)


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
        self.drop4 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.2)
        
        self.batch = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_1 = torch.nn.Linear(n_output, 3*n_output)
        self.linear_2 = torch.nn.Linear(3*n_output, n_output)
        self.linear_3 = torch.nn.LazyLinear(self.output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv1_1(x)
        x = F.relu(self.bn1_1(x))


        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.drop2(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.drop3(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool5(x)
        x = self.drop4(x)

        # x = self.conv6(x)
        # x = F.relu(self.bn5(x))
        # x = self.pool6(x)
        # x = self.drop4(x)

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
        return s
        
    
    def __init__(self, num_classes=10):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 32, 32)
        self.l2 = self.two_conv_pool(32, 64, 64)
        self.l3 = self.three_conv_pool(64, 128, 128, 128)
        # self.l4 = self.three_conv_pool(128, 128, 128, 128)
        
        Half_width = 64
        layer_width = 64
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
        Half_width = 64
        layer_width = 64
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # x = self.l4(x)
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


class MarbleNet(nn.Module):
  def __init__(self, num_classes, C=130):
    super(MarbleNet, self).__init__()
    dropout = 0.2
    self.prologue = nn.Sequential(
      nn.Conv1d(C//2, C//2, groups=65, kernel_size=11, padding='same', bias=False),
      nn.Conv1d(C//2, C, groups=1, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C),
      nn.ReLU(inplace=True)
    )

    self.sub00 = nn.Sequential(
      nn.Conv1d(C, C, kernel_size=13, groups=C, padding='same', bias=False),
      nn.Conv1d(C, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),

      nn.Conv1d(C//2, C//2, kernel_size=13, groups=C//2, padding='same', bias=False),
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2),
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

    self.sub22 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Dropout(dropout)
    )

    self.sub2C = nn.Sequential(
      nn.Conv1d(C//2, C//2, kernel_size=1, padding='same', bias=False),
      nn.BatchNorm1d(C//2)
    )

    self.epi1 = nn.Sequential(
      nn.Conv1d(C//2, C//2, groups=C//2, kernel_size=29, dilation=2, padding='same', bias=False),
      nn.Conv1d(C//2, C, kernel_size=1, padding='same', bias=False),
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
    
    x = x + x_
    x = self.sub02(x)

    x_ = self.sub1C(x)
    x = self.sub10(x)
    x = x + x_
    x = self.sub12(x)

    x_ = self.sub2C(x)
    x = self.sub20(x)
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
          mod_list.append(nn.Dropout(0))
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

if __name__ == '__main__':
  model = MarbleNet(10)
  tensor_ = torch.ones([10, 1, 65, 32])
  print(sum(p.numel() for p in model.parameters() if p.requires_grad))
