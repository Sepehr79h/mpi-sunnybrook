import torch.nn.functional as F
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(num_features=16)
        self.bn2 = nn.BatchNorm3d(num_features=32)
        self.bn3 = nn.BatchNorm3d(num_features=64)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)
        self.do = nn.Dropout3d(p=0.5)
        self.l1 = nn.Linear(128, 1024)
        self.l2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, stat_features=None):
        x = self.conv_layers1(x)
        y = self.conv_layers1(y)
        x = self.conv_layers2(x)
        y = self.conv_layers2(y)
        x = self.conv_layers3(x)
        y = self.conv_layers3(y)
        #breakpoint()
        N, _, _, _, _ = x.size()
        x = x.view(N, -1)
        y = y.view(N, -1)
        # z = self.conv_layers2(x + y)
        z = torch.cat((x, y), 1)
        # z = z.view(N,-1)
        # print(z.size())
        #breakpoint()
        z = self.l1(z)
        z = self.relu(z)
        z = self.l2(z)
        z = self.sigmoid(z)
        return z

    def conv_layers1(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mp(x)
        return x

    def conv_layers2(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.mp(x)
        return x

    def conv_layers3(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.mp(x)
        return x