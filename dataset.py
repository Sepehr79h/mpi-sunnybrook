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
            # **self.labels[int(study_id)]  # unpack all labels into this sample
        }

        return sample
