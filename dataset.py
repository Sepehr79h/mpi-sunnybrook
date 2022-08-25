import re

import numpy as np
from torch.utils.data import Dataset
from data_analysis.visualize import plot_3d

class MPIDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        #breakpoint()
        study_id = self.labels.iloc[[idx]]["Study_ID"].iloc[0]
        sample_input = self.inputs[str(study_id)]
        #study_id = list(self.inputs.keys())[idx]
        #sample_input = self.inputs[study_id]

        sample = {
            "image": sample_input["series_images"],
            "image_list": sample_input["image_list"],
            "patient_sex": sample_input["PatientSex"],
            "patient_age": sample_input["PatientAge"],
            "impression": self.labels.loc[self.labels['Study_ID'] == int(study_id)]["Impression"].iloc[0]
        }

        # breakpoint()

        if self.transform:
            # breakpoint()
            # plot_3d(sample["image"])
            sample["image"] = self.transform(sample["image"])

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
