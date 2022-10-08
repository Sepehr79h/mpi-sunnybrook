import re

import numpy as np
import torch
from torch.utils.data import Dataset
from data_analysis.visualize import plot_3d


class MPIDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs[0]
        self.inputs_statistical = inputs[1]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        study_id = self.labels.iloc[[idx]]["Study_ID"].iloc[0]
        sample_input = self.inputs[str(study_id)]

        sample = {
            "image": sample_input["series_images"],
            "stat_features": self.inputs_statistical.loc[int(study_id)].values,
            "impression": self.labels.loc[self.labels['Study_ID'] == int(study_id)]["Impression"].iloc[0]
        }

        if self.transform:
            # breakpoint()
            # import matplotlib.pyplot as plt
            # plt.imshow(sample["image"][5, :, :], cmap='gray')
            # plt.show()

            sample["image"] = self.transform(sample["image"])
            sample["image"] = torch.permute(sample["image"], (1, 2, 0))

            # breakpoint()
            # print(sample["impression"])
            # import matplotlib.pyplot as plt
            # plt.imshow(sample["image"][5, :, :], cmap='gray')
            # plt.show()

        return sample


class MinMaxNormalize(object):
    def __init__(self, min_pixel_value, max_pixel_value, max_age):
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.max_age = max_age

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.min_pixel_value) / (self.max_pixel_value - self.min_pixel_value)
        sample["image"] = np.nan_to_num(sample["image"])
        sample["patient_age"] /= self.max_age
        return sample
