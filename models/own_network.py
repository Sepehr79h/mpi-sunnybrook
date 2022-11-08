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
        num_classes: int = 2,
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


class CSANet(nn.Module):
    """
    Category-based subspace attention network (CSA-Net)
    reference: https://arxiv.org/pdf/1912.08967.pdf
    """

    def __init__(self, num_subspaces=5, embedding_size=64):
        """
        :param num_subspaces: (int) number of subspaces that an image can be in
        :param embedding_size: (int) dimension of embedding feature
        """
        super(CSANet, self).__init__()
        self.num_subspaces = num_subspaces
        self.embedding_size = embedding_size
        # we use reset18 as per the paper
        weights_path = "/mnt/5gb_ssd/sepehr/Repos/mpi-sunnybrook/models/pretrain_weights/resnet_10_23dataset.pth"
        # self.resnet10 = create_pretrained_medical_resnet(weights_path)
        self.resnet18 = generate_model(10, n_input_channels=1)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-6])
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10+7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, str_image, rst_image, x_stats):

        str_feature = self.resnet18(str_image)
        rst_feature = self.resnet18(rst_image)
        out = torch.cat((str_feature.reshape(str_feature.shape[0], -1), rst_feature.reshape(str_feature.shape[0], -1)),-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.cat((out, x_stats), dim=-1)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out


class AttentionLayer(nn.Module):
    """
    The attention layer will calculate attention weights and
    combine those weights with features from CSA-Net to output final embedding result
    """

    def __init__(self, num_subspaces=5):
        super(AttentionLayer, self).__init__()
        self.num_subspaces = num_subspaces
        # two fc layers as per the paper
        # TODO: removes hardcoded dimension in the first fc layer
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, self.num_subspaces)
        # init them
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, feature, item_category, target_category):
        """
        :param feature: (tensor) image features extracted from CSA-Net (dim=64)
        :param item_category: (one-hot tensors) categories of source item
        :param target_category: (one-hot tensors) categories of item that we want to predict compatibility
        :return: (tensor) embedding of item in the subspace of source and target category
        """
        # we usually in a situation when there is only one item category vs multiple target categories and vice versa
        # so we have to stack the one that have smaller shape to make them equal in term of shape
        # TODO: find a better way to deal with this situation
        if len(item_category.shape) > len(target_category.shape):
            target_category = target_category.repeat(item_category.shape[0], 1)
        elif len(item_category.shape) < len(target_category.shape):
            item_category = item_category.repeat(target_category.shape[0], 1)

        # same thing happens with feature
        if feature.shape[0] < item_category.shape[0]:
            feature = feature.repeat(item_category.shape[0], 1, 1)

        # combied_category = torch.cat((item_category, target_category), 1)
        attention_weights = self.fc1(torch.cat((item_category, target_category), 1))
        attention_weights = self.fc2(attention_weights)
        attention_weights = nn.functional.softmax(attention_weights)
        attention_weights = attention_weights.unsqueeze(-1)
        feature = feature * attention_weights
        embedding = torch.sum(feature, dim=1)

        return embedding
