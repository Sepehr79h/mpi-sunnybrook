# import PyTorch
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, stride=2)
        self.pool = nn.MaxPool3d(2, stride=2)
        self.conv2 = nn.Conv3d(16, 16, 3, stride=2, padding=2)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=2)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=128, bidirectional=False, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64, bidirectional=False, batch_first=True)
        self.lstm_3 = nn.LSTM(input_size=64, hidden_size=64, bidirectional=False, batch_first=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input, x_stats=None):
        output = self.conv1(input)
        output = self.relu(self.pool(output))
        output = self.conv2(output)
        output = self.relu(self.pool(output))
        # output = self.conv3(output)
        # output = self.relu(self.pool(output))
        #breakpoint()
        #output = output.view(tuple((-1, *output.shape[:-2])))
        #breakpoint()

        output = torch.flatten(output, start_dim=1)
        #breakpoint()

        # output = output.reshape((output.shape[0], output.shape[1], output.shape[2]*output.shape[3]*output.shape[4]))
        # #breakpoint()
        # output, _ = self.lstm_1(output)
        # output = self.relu(output)
        # output, _ = self.lstm_2(output)
        # output = self.relu(output)
        # _, (output, c_n) = self.lstm_3(output)


        #output = self.relu(output)
        #output = self.relu(self.linear1(output))
        #output = self.relu(self.linear2(output))
        #output = self.sigmoid(self.linear3(output)).squeeze(0)
        return output.squeeze(0)