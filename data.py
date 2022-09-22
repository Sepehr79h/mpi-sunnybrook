import copy
import re

import imageio
import numpy as np
import torch
import os
import pydicom
import pickle
import time
import pandas as pd
from skimage.morphology.tests.test_gray import im
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import WeightedRandomSampler

import global_vars as GLOBALS

from torch.utils.data.dataset import random_split, Subset
from dataset import MPIDataset
from dataset import MinMaxNormalize
from tqdm import tqdm
from torchvision import transforms
from data_analysis.visualize import plot_3d
from sklearn import preprocessing


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
    frame_size_list, height_list, width_list = [], [], []

    for station_name in station_info.keys():

        raw_data = station_data_dict[station_name]
        desired_image_patterns = station_info[station_name]

        for study_id in raw_data.keys():
            images = raw_data[study_id]["series_images"]
            for i in range(len(desired_image_patterns)):
                match = re.search(desired_image_patterns[i], "".join(images.keys()))
                if match:
                    image_name = match.group()

                    frame_size_list += [images[image_name].shape[0]]
                    width_list += [images[image_name].shape[1]]
                    height_list += [images[image_name].shape[2]]

                    max_image_frames[i] = max(max_image_frames[i], images[image_name].shape[0])
                    max_image_width[i] = max(max_image_width[i], images[image_name].shape[1])
                    max_image_height[i] = max(max_image_height[i], images[image_name].shape[2])

    # import matplotlib.pyplot as plt
    # plt.hist(frame_size_list, bins=np.arange(min(frame_size_list), max(frame_size_list) + 2, 1), ec='black')
    # plt.xlabel("Frame Size")
    # plt.ylabel("Num Images")
    # plt.show()
    # plt.hist(width_list, bins=np.arange(min(width_list), max(width_list) + 2, 1), ec='black')
    # plt.xlabel("Image Width")
    # plt.ylabel("Num Images")
    # plt.show()
    # plt.xlabel("Image Height")
    # plt.ylabel("Num Images")
    # plt.hist(height_list, bins=np.arange(min(height_list), max(height_list) + 2, 1), ec='black')
    # plt.show()
    # breakpoint()
    return max_image_frames, max_image_height, max_image_width


def clean_data(station_data_dict, labels, station_info):
    data = copy.deepcopy(station_data_dict)
    class_count = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    }
    count = 0
    for station_name in station_info.keys():
        raw_inputs = station_data_dict[station_name]
        desired_image_patterns = station_info[station_name]
        combined_pattern = "^(?=.*" + ")(?=.*".join(desired_image_patterns) + ")"

        for study_id in raw_inputs.keys():

            images = raw_inputs[study_id]["series_images"]
            patient_sex = raw_inputs[study_id]["PatientSex"]
            patient_age = raw_inputs[study_id]["PatientAge"]

            df_patient = labels.loc[labels['Study_ID'] == int(study_id)]
            impression = df_patient["Impression"]
            #stress_test_features = df_patient.drop(['Impression', 'Study_ID'], axis=1)
            # breakpoint()

            # Clean Data
            if not re.search(rf"{combined_pattern}", "".join(images.keys())) \
                    or not patient_age or not patient_sex \
                    or impression.empty or set() == set(GLOBALS.CONFIG["classes"]).intersection(
                set(impression.unique())) \
                    or class_count[str(impression.unique()[0])] > GLOBALS.CONFIG["max_images_per_class"]:
                data[station_name].pop(study_id)
                count += 1
                continue

            for i in range(len(desired_image_patterns)):
                if GLOBALS.CONFIG["max_frame_size"]:
                    image_name = re.search(desired_image_patterns[i], "".join(images.keys())).group()
                    if images[image_name].shape[0] > GLOBALS.CONFIG["max_frame_size"]:
                        count += 1
                        data[station_name].pop(study_id)
                        break
            else:
                class_count[str(impression.unique()[0])] += 1

    print(f"Filtered {count} examples during data cleaning.")
    return data


def process_data(station_data_dict, labels, station_info):
    inputs = {}
    targets = pd.DataFrame()
    total_pixels, sum_x, sum_x_sq = 0, 0.0, 0.0
    data_stats = {
        "min_pix": 0,
        "max_pix": 0,
        "max_age": 0,
        "mean": 0
    }

    station_data_dict_clean = clean_data(station_data_dict, labels, station_info)
    max_image_frames, max_image_width, max_image_height = get_max_image_frames(station_data_dict_clean, station_info)

    for station_name in station_info.keys():
        raw_inputs = station_data_dict_clean[station_name]
        desired_image_patterns = station_info[station_name]
        processed_data = copy.deepcopy(raw_inputs)

        for study_id in raw_inputs.keys():

            images = raw_inputs[study_id]["series_images"]
            patient_sex = raw_inputs[study_id]["PatientSex"]
            patient_age = raw_inputs[study_id]["PatientAge"]
            targets = pd.concat([targets, labels.loc[labels['Study_ID'] == int(study_id)]], ignore_index=True)

            # Process Data
            stacked_images = []
            for i in range(len(desired_image_patterns)):
                image_name = re.search(desired_image_patterns[i], "".join(images.keys())).group()

                total_pixels += images[image_name].size
                sum_x += np.sum(images[image_name])
                sum_x_sq += np.sum(np.square(images[image_name]))

                if GLOBALS.CONFIG["max_frame_size"]:
                    pad_size_frames = max(0, GLOBALS.CONFIG["max_frame_size"] - images[image_name].shape[0])
                else:
                    pad_size_frames = max_image_frames[i] - images[image_name].shape[0]

                pad_size_width = max_image_width[i] - images[image_name].shape[1]
                pad_size_height = max_image_height[i] - images[image_name].shape[2]
                padded_image = np.pad(images[image_name],
                                      ((0, pad_size_frames), (0, pad_size_width), (0, pad_size_height)))
                padded_image = padded_image[:,
                               GLOBALS.CONFIG['height_crop_size']:padded_image.shape[1] - GLOBALS.CONFIG[
                                   'height_crop_size'],
                               GLOBALS.CONFIG['width_crop_size']:padded_image.shape[2] - GLOBALS.CONFIG[
                                   'width_crop_size']]
                # padded_image = padded_image[0: GLOBALS.CONFIG['max_frame_size'],
                #                GLOBALS.CONFIG['height_crop_size']:padded_image.shape[1] - GLOBALS.CONFIG[
                #                    'height_crop_size'],
                #                GLOBALS.CONFIG['width_crop_size']:padded_image.shape[2] - GLOBALS.CONFIG[
                #                    'width_crop_size']]
                stacked_images.append(padded_image.astype(np.float))

                # import matplotlib.pyplot as plt
                # plt.imshow(padded_image[5, :, :], cmap='gray')
                # plt.show()
                # breakpoint()

                # print(images[image_name].shape)
                # breakpoint()
                # images[image_name] = images[image_name][:, 20:44:, 5:59]
                # import matplotlib.pyplot as plt
                # plt.imshow(images[image_name][5, :, :], cmap='gray')
                # plt.show()
                # breakpoint()

                # for j in range(0,images[image_name].shape[0]):
                #     plt.subplot(2, images[image_name].shape[0]//2, j+1)
                #     plt.imshow(images[image_name][j, :, :], cmap='gray')
                #
                # plt.show()
                #
                # breakpoint()

            processed_data[study_id]["image_list"] = stacked_images
            processed_data[study_id]["series_images"] = np.vstack(stacked_images)
            processed_data[study_id]["PatientAge"] = int(patient_age[:-1]) / 100
            processed_data[study_id]["PatientSex"] = 0 if patient_sex == "M" else 1

            data_stats["max_age"] = max(data_stats["max_age"], processed_data[study_id]["PatientAge"])

            # plot_3d(processed_data[study_id]["series_images"])
            # breakpoint()

        inputs.update(processed_data)

    data_stats["mean"] = sum_x / total_pixels
    data_stats["std"] = np.sqrt(sum_x_sq / total_pixels + data_stats["mean"] ** 2)

    le = preprocessing.LabelEncoder()
    targets['Impression'] = le.fit_transform(targets['Impression'])

    # plot_sample_image(inputs["8718"]["series_images"], 10, 6)
    # breakpoint()

    return inputs, targets, data_stats


def plot_sample_image(image, subplot_rows, subplot_height):
    import matplotlib.pyplot as plt
    for j in range(0, image.shape[0]):
        plt.subplot(subplot_rows, subplot_height, j + 1)
        plt.imshow(image[j, :, :], cmap='gray')
    plt.show()


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

    inputs, labels, data_stats = process_data(station_data_dict, labels, station_info)

    print("Data loaded successfully.")

    # transform = transforms.Compose(
    #     [MinMaxNormalize(data_stats["min_pix"], data_stats["max_pix"], data_stats["max_age"])])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(data_stats["mean"], data_stats["std"]),
         # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
         transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))])
    print(f"Transforms: {transform}")

    dataset = MPIDataset(inputs, labels, transform=transform)
    labels_unique, counts = np.unique(labels["Impression"], return_counts=True)
    print(f"Dataset Size: {len(dataset)}")
    print(f"Labels: {labels_unique}, Distribution: {counts}")

    train_dataset, test_dataset = get_dataset_splits(labels, dataset)
    sampler = get_sampler(train_dataset, labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=GLOBALS.CONFIG["batch_size"], shuffle=False,
                                               sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=GLOBALS.CONFIG["batch_size"], shuffle=False)

    return train_loader, test_loader


def get_dataset_splits(labels, dataset):
    if GLOBALS.CONFIG["split_type"] == "balanced":
        targets = labels["Impression"].to_numpy()
        train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), stratify=targets,
                                                       train_size=GLOBALS.CONFIG["train_split_percentage"])
        _, train_counts = np.unique(targets[train_indices], return_counts=True)
        _, test_counts = np.unique(targets[test_indices], return_counts=True)
        print(f"Train Class Distribution: {train_counts}")
        print(f"Test Class Distribution: {test_counts}")

        train_dataset = Subset(dataset, indices=train_indices)
        test_dataset = Subset(dataset, indices=test_indices)
        return train_dataset, test_dataset

    elif GLOBALS.CONFIG["split_type"] == "random":
        train_split = int(len(dataset) * GLOBALS.CONFIG["train_split_percentage"])
        test_split = len(dataset) - train_split
        train_dataset, test_dataset = random_split(dataset, [train_split, test_split])
        return train_dataset, test_dataset


def get_sampler(train_dataset, labels):
    sampler = None
    if GLOBALS.CONFIG["sampler"] == "WeightedRandomSampler":
        y_train_indices = train_dataset.indices
        y_train = [labels.iloc[[i]]["Impression"].iloc[0] for i in y_train_indices]
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler
