import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_classes=4, num_covariates=2):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        # self.fc1 = nn.Linear(14 * 14 * 28 * 64, 128)
        # self.fc1 = nn.Linear(14*14*59*64, 128)
        self.fc1 = nn.Linear(15 * 15 * 19 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)
        self.softmax = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(num_classes+num_covariates, num_classes)
        #self.fc3 = nn.Linear(num_classes+num_covariates, 128)
        #self.fc4 = nn.Linear(128, num_classes)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x_image, x_stats=None):
        # Set 1
        out = self.conv_layer1(x_image)
        out = self.conv_layer2(out)
        #breakpoint()
        out = out.view(out.size(0), -1)
        # out = self.drop(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        #out = self.drop(out)
        out = self.fc2(out)
        #out = self.relu(out)
        #out = self.softmax(out)

        #out = torch.cat((out, torch.stack(x_stats, dim=1)), dim=1)
        #out = self.fc3(out)

        #out = self.relu(out)
        #out = self.batch(out)
        #out = self.relu(out)
        #out = self.fc4(out)

        return out
