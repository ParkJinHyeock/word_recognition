import torch
torch.manual_seed(215)
import torch.nn as nn
import torch.nn.functional as F



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
        self.drop1 = nn.Dropout2d(p=0.1)
        self.drop2 = nn.Dropout2d(p=0.1)
        self.drop3 = nn.Dropout(p=0.2)
        self.batch = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_1 = torch.nn.Linear(n_output, 3*n_output)
        self.linear_2 = torch.nn.Linear(3*n_output, n_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.drop1(x)

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

        if self.weight is None:
           self.weight = nn.Parameter(torch.randn(self.output, x.size()[1])).to(self.device)
        x = F.linear(x, self.weight)

        return F.log_softmax(x, dim=1)