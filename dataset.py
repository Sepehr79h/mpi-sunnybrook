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

        images = sample_input["series_images"]
        sample = {
            "images": np.concatenate((
                images[re.search("RST._U_TF_SD._SA", "".join(images.keys())).group()],
                images[re.search("RST._S_TF_SD._SA", "".join(images.keys())).group()],
                images[re.search("STR._U_TF_SD._SA", "".join(images.keys())).group()],
                images[re.search("STR._S_TF_SD._SA", "".join(images.keys())).group()],
            )).astype(np.float),
            "PatientSex": sample_input["PatientSex"],
            "PatientAge": sample_input["PatientAge"],
            # **self.labels[int(study_id)]  # unpack all labels into this sample
        }

        breakpoint()

        return sample