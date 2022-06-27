import re

import numpy as np
import torch
import os
import pydicom
import pickle
import time
import pandas as pd
import global_vars as GLOBALS

from torch.utils.data.dataset import random_split
from dataset import MPIDataset
from tqdm import tqdm


def save_dicom_store(dicom_path, station_data_path, station_name):
    data = {}

    time_start = time.time()

    for root, dirs, files in tqdm(os.walk(dicom_path)):

        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext == ".dcm":
                ds = pydicom.dcmread(os.path.join(root, file))
                if ds.StationName == station_name:
                    if ds.PatientID not in data:
                        data[ds.PatientID] = {
                            "series_images": {ds.SeriesDescription: ds.pixel_array},
                            "PatientSex": ds.PatientSex,
                            "PatientAge": ds.PatientAge
                        }
                    else:
                        data[ds.PatientID]["series_images"][ds.SeriesDescription] = ds.pixel_array

    with open(os.path.join(station_data_path, f"dicom_{station_name}.pkl"), 'wb') as f:
        pickle.dump(data, f)

    time_end = time.time()

    print("Dicom data saved. Time taken: {:.2f} s".format(time_end - time_start))


def get_max_image_frames(raw_data, desired_image_patterns):
    max_image_frames = {}
    for study_id in raw_data.keys():
        images = raw_data[study_id]["series_images"]
        for pattern in desired_image_patterns:
            match = re.search(pattern, "".join(images.keys()))
            if match:
                image_name = match.group()
                curr_max = max_image_frames[pattern] if pattern in max_image_frames.keys() else 0
                max_image_frames[pattern] = max(curr_max, len(images[image_name]))

    return max_image_frames


def process_data(raw_inputs, labels):
    processed_data = raw_inputs.copy()
    desired_image_patterns = [
        "RST._U_TF_SD._SA",
        "RST._S_TF_SD._SA",
        "STR._U_TF_SD._SA",
        "STR._S_TF_SD._SA"
    ]
    combined_pattern = "^(?=.*" + ")(?=.*".join(desired_image_patterns) + ")"
    max_image_frames = get_max_image_frames(raw_inputs, desired_image_patterns)

    for study_id in raw_inputs.keys():

        images = raw_inputs[study_id]["series_images"]
        patient_sex = raw_inputs[study_id]["PatientSex"]
        patient_age = raw_inputs[study_id]["PatientAge"]
        impression = labels.loc[labels['Study_ID'] == int(study_id)]["Impression"]

        # Clean Data
        if not re.search(rf"{combined_pattern}", "".join(images.keys())) \
                or not patient_age or not patient_sex \
                or impression.empty or (5 in impression.unique()):
            processed_data.pop(study_id)
            continue

        # Process Data
        stacked_images = []
        for pattern in desired_image_patterns:
            image_name = re.search(pattern, "".join(images.keys())).group()
            pad_size = max_image_frames[pattern] - len(images[image_name])
            padded_image = np.pad(images[image_name], ((0, pad_size), (0, 0), (0, 0)))
            stacked_images.append(padded_image)

        processed_data[study_id]["series_images"] = np.vstack(stacked_images)
        processed_data[study_id]["PatientAge"] = int(patient_age[:-1]) / 100
        processed_data[study_id]["PatientSex"] = 0 if patient_sex == "M" else 1

        # Normalize Data
        # v_min = processed_data[study_id]["series_images"].min(axis=(1, 2), keepdims=True)
        # v_max = processed_data[study_id]["series_images"].max(axis=(1, 2), keepdims=True)
        # processed_data[study_id]["series_images"] = (processed_data[study_id]["series_images"] - v_min)/(v_max - v_min)
        # processed_data[study_id]["series_images"] = np.nan_to_num(processed_data[study_id]["series_images"])

    return processed_data


def load_data(dicom_path, labels_path, station_data_path, station_name):
    print(f"Fetching dicom store from path: {dicom_path}")
    print(f"Fetching labels from path: {labels_path}")

    if not os.path.exists(os.path.join(station_data_path, f"dicom_{station_name}.pkl")):
        # create data object holding all data, save to file for fast loading
        save_dicom_store(dicom_path, station_data_path, station_name)

    with open(os.path.join(station_data_path, f"dicom_{station_name}.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    labels = pd.read_excel(labels_path, sheet_name="Data")
    inputs = process_data(raw_data, labels)

    print(f"Data loaded successfully from station: {station_name}")

    dataset = MPIDataset(inputs, labels)

    train_split = int(len(dataset) * GLOBALS.CONFIG["train_split_percentage"])
    test_split = len(dataset) - train_split
    train_dataset, test_dataset = random_split(dataset, [train_split, test_split])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=GLOBALS.CONFIG["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=GLOBALS.CONFIG["batch_size"], shuffle=False)

    return train_loader, test_loader


def load_data_sample():
    # temporary sample data

    batch_size = 2

    train_x = torch.rand(50, 40, 64, 64)  # num images, num channels, img height, img width
    train_x = torch.unsqueeze(train_x, dim=1)
    test_x = torch.rand(10, 40, 64, 64)

    train_y = torch.randint(0, 4, (50,))
    test_y = torch.randint(0, 4, (10,))

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
