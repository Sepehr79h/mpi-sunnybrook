from typing import Any, Tuple

import torch.nn as nn
import torchvision.models as models
import torch

from models.resnet_3d import generate_model
# from models.monai_resnet import resnet10

from monai.networks.nets import EfficientNetBN, ResNet, resnet10
from models.lstm import LSTM


def create_pretrained_medical_resnet(
        pretrained_path: str,
        model_constructor: callable = resnet10,
        spatial_dims: int = 3,
        n_input_channels: int = 1,
        num_classes: int = 1,
        **kwargs_monai_resnet: Any):
    """This si specific constructor for MONAI ResNet module loading MedicalNEt weights.
    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['state_dict'], strict=False)
    return net


class TransferModel(nn.Module):
    """
    Category-based subspace attention network (CSA-Net)
    reference: https://arxiv.org/pdf/1912.08967.pdf
    """

    def __init__(self, num_subspaces=5, embedding_size=64):
        """
        :param num_subspaces: (int) number of subspaces that an image can be in
        :param embedding_size: (int) dimension of embedding feature
        """
        super(TransferModel, self).__init__()
        self.num_subspaces = num_subspaces
        self.embedding_size = embedding_size
        # we use reset18 as per the paper
        # weights_path = "/mnt/5gb_ssd/sepehr/Repos/mpi-sunnybrook/models/pretrain_weights/resnet_10_23dataset.pth"
        # self.medical_resnet = create_pretrained_medical_resnet(weights_path)
        # self.medical_resnet = nn.Sequential(*list(self.medical_resnet.children())[:-1])
        # breakpoint()
        self.resnet18 = generate_model(10, n_input_channels=1)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-6])
        # breakpoint()
        self.fc1 = nn.Linear(1024, 10)
        #self.fc1 = nn.Linear(512, 128)


        # self.bn1 = nn.BatchNorm1d(128)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # # self.fc2 = nn.Linear(128, 10)
        # # self.fc3 = nn.Linear(10+7, 1)
        # self.fc2 = nn.Linear(128, 10)


        self.fc3 = nn.Linear(10 + 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, str_s_image, str_u_image, rst_s_image, rst_u_image, x_stats):
        # breakpoint()
        str_s_feature = self.resnet18(str_s_image)
        str_u_feature = self.resnet18(str_u_image)
        rst_s_feature = self.resnet18(rst_s_image)
        rst_u_feature = self.resnet18(rst_u_image)

        out = torch.cat((str_s_feature.reshape(str_s_feature.shape[0], -1),
                         str_u_feature.reshape(str_u_feature.shape[0], -1),
                         rst_s_feature.reshape(rst_s_feature.shape[0], -1),
                         rst_u_feature.reshape(rst_u_feature.shape[0], -1)), -1)

        # breakpoint()
        # out = torch.cat((str_feature.reshape(str_feature.shape[0], -1), rst_feature.reshape(str_feature.shape[0], -1)),-1)
        out = self.fc1(out)
        #out = self.relu(out)
        #out = self.fc2(out)
        out = torch.cat((out, x_stats), dim=-1)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out
