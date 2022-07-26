import re

import numpy as np
from torch.utils.data import Dataset


class MPIDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        study_id = list(self.inputs.keys())[idx]
        sample_input = self.inputs[study_id]

        sample = {
            "image": sample_input["series_images"].astype(np.float),
            "patient_sex": sample_input["PatientSex"],
            "patient_age": sample_input["PatientAge"],
            "impression": self.labels.loc[self.labels['Study_ID'] == int(study_id)]["Impression"].iloc[0] - 1
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class MinMaxNormalize(object):
    def __init__(self, min_pixel_value, max_pixel_value, max_age):
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.max_age = max_age

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.min_pixel_value) / (self.max_pixel_value - self.min_pixel_value)
        sample["patient_age"] /= self.max_age
        return sample
