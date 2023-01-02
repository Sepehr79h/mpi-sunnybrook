import re

import numpy as np
import torch
from torch.utils.data import Dataset
from data_analysis.visualize import plot_3d
import global_vars as GLOBALS


class MPIDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs[0]
        self.inputs_statistical = inputs[1]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        study_id = self.labels.iloc[idx]["Study_ID"]
        sample_input = self.inputs[str(study_id)]

        sample = {
            "image": sample_input["series_images"],
            "stress_image": sample_input["stress_image"],
            "rest_image": sample_input["rest_image"],
            "image_list": sample_input["image_list"],
            "stat_features": self.inputs_statistical.loc[int(study_id)].values,
            "impression": self.labels.loc[self.labels['Study_ID'] == int(study_id)]["Impression"].iloc[0]
        }

        if self.transform:
            # breakpoint()
            # print(sample["impression"])
            # import matplotlib.pyplot as plt
            # plt.imshow(sample["image"][5, :, :], cmap='gray')
            # plt.show()
            # breakpoint()
            # import matplotlib.pyplot as plt
            # for j in range(0, sample["image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(sample["image"][j, :, :], cmap='gray')
            # plt.show()
            # breakpoint()

            # if GLOBALS.CONFIG["model"] == "own_network":
            #     for i in range(0, len(sample["image_list"])):
            #         if torch.is_tensor(sample["image_list"][i]):
            #             sample["image_list"][i] = sample["image_list"][i].detach().cpu().numpy()
            #         sample["image_list"][i] = self.transform(sample["image_list"][i])
            #         sample["image_list"][i] = torch.permute(sample["image_list"][i], (1, 2, 0))
            #
            #         # import matplotlib.pyplot as plt
            #         # for j in range(0, sample["image_list"][i].shape[0]):
            #         #     plt.subplot(10, 10, j + 1)
            #         #     plt.imshow(sample["image_list"][i][j, :, :], cmap='gray')
            #         # plt.show()
            #         # breakpoint()
            #
            # else:
            # import matplotlib.pyplot as plt
            # for j in range(0, sample["image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(sample["image"][j, :, :], cmap='gray')
            # plt.show()
            # breakpoint()

            sample["image"] = np.transpose(sample["image"], (1, 2, 0))
            sample["image"] = self.transform(sample["image"])

            sample["stress_image"] = np.transpose(sample["stress_image"], (1, 2, 0))
            sample["stress_image"] = self.transform(sample["stress_image"])
            #sample["stress_image"] = torch.permute(sample["stress_image"], (1, 2, 0))

            sample["rest_image"] = np.transpose(sample["rest_image"], (1, 2, 0))
            sample["rest_image"] = self.transform(sample["rest_image"])
            #sample["rest_image"] = torch.permute(sample["rest_image"], (1, 2, 0))

            # # breakpoint()
            # import matplotlib.pyplot as plt
            # for j in range(0, sample["image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(sample["image"][j, :, :], cmap='gray')
            # plt.show()
            # breakpoint()
            # breakpoint()
            # import matplotlib.pyplot as plt
            # for j in range(0, sample["stress_image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(sample["stress_image"][j, :, :], cmap='gray')
            # plt.show()
            # for j in range(0, sample["rest_image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(sample["rest_image"][j, :, :], cmap='gray')
            # plt.show()
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
