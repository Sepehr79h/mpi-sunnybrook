import torch
from torch import nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(96, 256, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout(p=0.3),

            nn.Conv3d(256, 384, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(384, 256, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(12800, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        x = torch.unsqueeze(x, dim=1)
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
