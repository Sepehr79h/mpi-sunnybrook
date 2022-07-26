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
from data_analysis.visualize import plot_3d


def save_dicom_store(dicom_path, station_data_path, station_name, series_descriptions):
    data = {}

    time_start = time.time()

    for root, dirs, files in tqdm(os.walk(dicom_path)):

        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext == ".dcm":
                try:
                    ds = pydicom.dcmread(os.path.join(root, file))
                    if ds.StationName == station_name and ds.SeriesDescription in series_descriptions:
                        if ds.PatientID not in data:
                            data[ds.PatientID] = {
                                "series_images": {ds.SeriesDescription: ds.pixel_array},
                                "PatientSex": ds.PatientSex,
                                "PatientAge": ds.PatientAge
                            }
                        else:
                            data[ds.PatientID]["series_images"][ds.SeriesDescription] = ds.pixel_array
                except:
                    print("Cannot read data from file. Skipping file.")

    with open(os.path.join(station_data_path, f"dicom_{station_name}.pkl"), 'wb') as f:
        pickle.dump(data, f)

    time_end = time.time()

    print("Dicom data saved for station: {}. Time taken: {:.2f} s".format(station_name, time_end - time_start))


def get_max_image_frames(station_data_dict, station_info):
    num_patterns = len(next(iter(station_info.values())))
    max_image_frames, max_image_height, max_image_width = [0] * num_patterns, [0] * num_patterns, [0] * num_patterns

    for station_name in station_info.keys():

        raw_data = station_data_dict[station_name]
        desired_image_patterns = station_info[station_name]

        for study_id in raw_data.keys():
            images = raw_data[study_id]["series_images"]
            for i in range(len(desired_image_patterns)):
                match = re.search(desired_image_patterns[i], "".join(images.keys()))
                if match:
                    image_name = match.group()
                    max_image_frames[i] = max(max_image_frames[i], images[image_name].shape[0])
                    max_image_width[i] = max(max_image_width[i], images[image_name].shape[1])
                    max_image_height[i] = max(max_image_height[i], images[image_name].shape[2])

    return max_image_frames, max_image_height, max_image_width


def process_data(station_data_dict, labels, station_info):
    inputs = {}
    max_image_frames, max_image_width, max_image_height = get_max_image_frames(station_data_dict, station_info)

    for station_name in station_info.keys():
        raw_inputs = station_data_dict[station_name]
        desired_image_patterns = station_info[station_name]

        processed_data = raw_inputs.copy()
        combined_pattern = "^(?=.*" + ")(?=.*".join(desired_image_patterns) + ")"

        for study_id in raw_inputs.keys():

            images = raw_inputs[study_id]["series_images"]
            patient_sex = raw_inputs[study_id]["PatientSex"]
            patient_age = raw_inputs[study_id]["PatientAge"]
            impression = labels.loc[labels['Study_ID'] == int(study_id)]["Impression"]

            # Clean Data
            if not re.search(rf"{combined_pattern}", "".join(images.keys())) \
                    or not patient_age or not patient_sex \
                    or impression.empty or set() == {1, 2, 3, 4}.intersection(set(impression.unique())):
                processed_data.pop(study_id)
                continue

            # Process Data
            stacked_images = []
            for i in range(len(desired_image_patterns)):
                image_name = re.search(desired_image_patterns[i], "".join(images.keys())).group()
                pad_size_frames = max_image_frames[i] - images[image_name].shape[0]
                pad_size_width = max_image_width[i] - images[image_name].shape[1]
                pad_size_height = max_image_height[i] - images[image_name].shape[2]
                padded_image = np.pad(images[image_name],
                                      ((0, pad_size_frames), (0, pad_size_width), (0, pad_size_height)))
                stacked_images.append(padded_image)

            processed_data[study_id]["series_images"] = np.vstack(stacked_images)
            processed_data[study_id]["PatientAge"] = int(patient_age[:-1]) / 100
            processed_data[study_id]["PatientSex"] = 0 if patient_sex == "M" else 1

            # Normalize Data
            v_min = processed_data[study_id]["series_images"].min(axis=(1, 2), keepdims=True)
            v_max = processed_data[study_id]["series_images"].max(axis=(1, 2), keepdims=True)
            processed_data[study_id]["series_images"] = (processed_data[study_id]["series_images"] - v_min)/(v_max - v_min)
            processed_data[study_id]["series_images"] = np.nan_to_num(processed_data[study_id]["series_images"])

        inputs.update(processed_data)

    return inputs


def load_data(dicom_path, labels_path, station_data_path, station_info):
    print(f"Fetching dicom store from path: {dicom_path}")
    print(f"Fetching labels from path: {labels_path}")

    labels = pd.read_excel(labels_path, sheet_name="Data")
    station_data_dict = {}

    for station_name in station_info.keys():
        if not os.path.exists(os.path.join(station_data_path, f"dicom_{station_name}.pkl")):
            # create data object holding all data, save to file for fast loading
            save_dicom_store(dicom_path, station_data_path, station_name, station_info[station_name])

        with open(os.path.join(station_data_path, f"dicom_{station_name}.pkl"), 'rb') as f:
            station_data_raw = pickle.load(f)
            station_data_dict[station_name] = station_data_raw

    inputs = process_data(station_data_dict, labels, station_info)

    print("Data loaded successfully.")

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
