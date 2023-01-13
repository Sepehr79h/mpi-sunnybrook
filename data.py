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
import global_vars as GLOBALS

from torch.utils.data.dataset import random_split, Subset
from dataset import MPIDataset
from dataset import MinMaxNormalize
from tqdm import tqdm
from torchvision import transforms
from data_analysis.visualize import plot_3d
from sklearn import preprocessing
from skimage.morphology.tests.test_gray import im
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import WeightedRandomSampler


def save_dicom_store(dicom_path, station_data_path, station_name, series_descriptions):
    """
    Description: This method saves the data for a specified station and its scans for fast data retrieval

    :param dicom_path: Path of dicom store
    :param station_data_path: Path to save dicom store
    :param station_name: Station name being saved
    :param series_descriptions: Scans to be saved from the station (e.g. RST and STR scans)
    """

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
    """
    Description: This method returns the max image frames, height, and width across all stations for each series description

    :param station_data_dict: Data object containing data for all stations specified in the config
    :param station_info: Names of all stations and their respective series descriptions specified in the config
    :return: max image frames, height, and width for each series description across all stations
    """
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

    return max_image_frames, max_image_height, max_image_width


def clean_data(station_data_dict, labels, station_info):
    """
    Description: This method removes any examples with missing or incorrect data from the dataset

    :param station_data_dict: Data object containing data for all stations specified in the config
    :param labels: Labels for all data in station_data_dict
    :param station_info: Names of all stations and their respective series descriptions specified in the config
    :return: The cleaned data_dict
    """
    data = copy.deepcopy(station_data_dict)
    class_count = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    }

    #count, count_frame_filter = 0, 0
    filter_counts = {
        "total": 0,
        "count_both_rst_str": 0,
        "count_patient_age": 0,
        "count_patient_sex": 0,
        "count_impression": 0,
        "count_lvef_stress_type": 0,
        "count_frame_filter": 0,
    }

    is_correct_type = True
    for station_name in station_info.keys():
        raw_inputs = station_data_dict[station_name]
        desired_image_patterns = station_info[station_name]
        combined_pattern = "^(?=.*" + ")(?=.*".join(desired_image_patterns) + ")"

        for study_id in raw_inputs.keys():

            images = raw_inputs[study_id]["series_images"]
            patient_sex = raw_inputs[study_id]["PatientSex"]
            patient_age = raw_inputs[study_id]["PatientAge"]

            df_patient = labels.loc[labels['Study_ID'] == int(study_id)]
            # df_patient = df_patient[["Study_ID", "Impression", "Stress Type"]]
            impression = df_patient["Impression"]

            #breakpoint()
            try:
                df_patient.astype(float)
                is_correct_type = True
            except:
                # print(df_patient)
                is_correct_type = False

            if not re.search(rf"{combined_pattern}", "".join(images.keys())):
                filter_counts["count_both_rst_str"] += 1
            elif not patient_age:
                filter_counts["count_patient_age"] += 1
            elif not patient_sex:
                filter_counts["count_patient_sex"] += 1
            elif impression.empty or set() == set(GLOBALS.CONFIG["classes"]).intersection(set(impression.unique())):
                filter_counts["count_impression"] += 1
            elif not is_correct_type or df_patient.isnull().values.any():
                filter_counts["count_lvef_stress_type"] += 1

            # Clean Data
            if not re.search(rf"{combined_pattern}", "".join(images.keys())) \
                    or not patient_age \
                    or not patient_sex \
                    or impression.empty \
                    or set() == set(GLOBALS.CONFIG["classes"]).intersection(set(impression.unique())) \
                    or class_count[str(impression.unique()[0])] > GLOBALS.CONFIG["max_images_per_class"] \
                    or not is_correct_type \
                    or df_patient.isnull().values.any():
                data[station_name].pop(study_id)
                filter_counts["total"] += 1
                continue

            for i in range(len(desired_image_patterns)):
                if GLOBALS.CONFIG["max_frame_size"]:
                    image_name = re.search(desired_image_patterns[i], "".join(images.keys())).group()
                    if images[image_name].shape[0] > GLOBALS.CONFIG["max_frame_size"]:
                        # print(station_name, images[image_name].shape)
                        filter_counts["count_frame_filter"] += 1
                        filter_counts["total"] += 1
                        data[station_name].pop(study_id)
                        break
            else:
                class_count[str(impression.unique()[0])] += 1

    print(f"Filter Counts: {filter_counts}")
    #print(f"Filtered {count} examples during data cleaning.")
    # breakpoint()
    return data


def get_df_categorical(df):
    """
    Description: This method processes the categorical attributes of the data

    :param df: Dataframe containing all categorical attributes to be used in the model
    :return: Returns the processed categorical dataframe to be used in the model
    """
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    df_categorical = pd.DataFrame(one_hot_encoder.fit_transform(df))
    df_categorical.index = df.index
    return df_categorical


def get_df_numerical(df):
    """
    Description: This method processes the numerical attributes of the data

    :param df: Dataframe containing all numerical attributes to be used in the model
    :return: Returns the processed numerical dataframe to be used in the model
    """
    df = df.astype(float)
    df_numerical = df[df.columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df_numerical


def process_data(station_data_dict, labels, station_info):
    """
    Description: This method performs all pre-processing for the data extracted from the di-com store

    :param station_data_dict: Data object containing data for all stations specified in the config
    :param labels: Labels for all data in station_data_dict
    :param station_info: Names of all stations and their respective series descriptions specified in the config
    :return: A tuple containing both image and stat data, labels, and important data stats
    """
    df_stat_features = pd.DataFrame()
    inputs = {}
    targets = pd.DataFrame()
    total_pixels, sum_x, sum_x_sq = 0, 0.0, 0.0
    data_stats = {
        "min_pix": 0,
        "max_pix": 0,
        "max_age": 0,
        "mean": 0
    }

    # Convert all label 4 patients (high risk) to label 3
    labels.loc[labels['Impression'] == 4, 'Impression'] = 3
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

            df_patient = labels.loc[labels['Study_ID'] == int(study_id)]
            df_stats_patient = df_patient.drop(['Impression'], axis=1)
            df_stats_patient = df_stats_patient.assign(patient_sex=patient_sex, patient_age=int(patient_age[:-1]))
            df_stat_features = pd.concat([df_stat_features, df_stats_patient])

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
                # breakpoint()
                stacked_images.append(padded_image.astype(np.float))

            # breakpoint()
            processed_data[study_id]["image_list"] = stacked_images
            processed_data[study_id]["series_images"] = np.vstack(stacked_images)
            processed_data[study_id]["stress_image"] = stacked_images[0]
            processed_data[study_id]["rest_image"] = stacked_images[1]

            # breakpoint()

            # import matplotlib.pyplot as plt
            # for j in range(0, processed_data[study_id]["stress_image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(processed_data[study_id]["stress_image"][j, :, :], cmap='gray')
            # plt.show()
            # for j in range(0, processed_data[study_id]["rest_image"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(processed_data[study_id]["rest_image"][j, :, :], cmap='gray')
            # plt.show()
            # breakpoint()

            # import matplotlib.pyplot as plt
            # for j in range(0, processed_data[study_id]["series_images"].shape[0]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(processed_data[study_id]["series_images"][j, :, :], cmap='gray')
            # plt.show()
            # breakpoint()

            # data_stats["max_age"] = max(data_stats["max_age"], processed_data[study_id]["PatientAge"])
            # plot_3d(processed_data[study_id]["series_images"])
            # breakpoint()

        inputs.update(processed_data)

    data_stats["mean"] = sum_x / total_pixels
    data_stats["std"] = np.sqrt(sum_x_sq / total_pixels + data_stats["mean"] ** 2)

    le = preprocessing.LabelEncoder()
    targets['Impression'] = le.fit_transform(targets['Impression'])

    numerical_features = GLOBALS.CONFIG["numerical_features"]
    categorical_features = GLOBALS.CONFIG["categorical_features"]
    print(f"Numerical features: {numerical_features}, Categorical Features: {categorical_features}")
    df_categorical = get_df_categorical(df_stat_features[categorical_features])
    df_numerical = get_df_numerical(df_stat_features[numerical_features])
    df_stat_features = pd.concat([df_stat_features["Study_ID"], df_categorical, df_numerical], axis=1)
    df_stat_features = df_stat_features.set_index('Study_ID')
    # breakpoint()

    # torch.from_numpy(df.values.astype(float))
    # plot_sample_image(inputs["8718"]["series_images"], 10, 6)

    return (inputs, df_stat_features), targets, data_stats


def plot_sample_image(image, subplot_rows, subplot_height):
    """
    Description: Plot a mpi image used in the dataset with all its corresponding frames

    :param image: The image to be plotted
    :param subplot_rows: Number of rows for subplots
    :param subplot_height: Number of columns for subplots
    """
    import matplotlib.pyplot as plt
    for j in range(0, image.shape[0]):
        plt.subplot(subplot_rows, subplot_height, j + 1)
        plt.imshow(image[j, :, :], cmap='gray')
    plt.show()


def load_data(dicom_path, labels_path, station_data_path, station_info):
    """
    Description: This method defines the dataset, train and test loaders, which will be ready to be used by the model

    :param dicom_path: Path of dicom store
    :param labels_path: Path of label excel file
    :param station_data_path: Path of the saved station data
    :param station_info: Names of all stations and their respective series descriptions specified in the config
    :return: The train and test loader to be used by the model
    """
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

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=30),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.Normalize(data_stats["mean"], data_stats["std"]),
            # transforms.RandomRotation(30),
            # transforms.RandomAffine(0, translate=(0.01, 0.01))
        ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(data_stats["mean"], data_stats["std"]),
         ])
    # ,
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))])
    print(f"Train Transforms: {transform_train}")

    full_dataset_train = MPIDataset(inputs, labels, transform=transform_train)
    full_dataset_test = MPIDataset(inputs, labels, transform=transform_test)

    labels_unique, counts = np.unique(labels["Impression"], return_counts=True)
    print(f"Dataset Size: {len(full_dataset_test)}")
    print(f"Labels: {labels_unique}, Distribution: {counts}")
    print(f"Image Input Shape: {list(inputs[0].values())[0]['series_images'].shape}")

    train_dataset, test_dataset = get_dataset_splits(labels, full_dataset_train, full_dataset_test)

    sampler = get_sampler(train_dataset, labels)
    weights = get_class_weights(train_dataset, labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=GLOBALS.CONFIG["batch_size"], shuffle=False,
                                               sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=GLOBALS.CONFIG["batch_size"], shuffle=False)

    return train_loader, test_loader, weights


def get_dataset_splits(labels, dataset_train, dataset_test):
    """
    Description: This method returns the dataset split depending on the specification in the config

    :param labels: Labels of the dataset
    :param dataset: The dataset containing all train and test data
    :return: The split train and test datasets to be used by the train and test loaders
    """
    if GLOBALS.CONFIG["split_type"] == "balanced":
        targets = labels["Impression"].to_numpy()
        train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), stratify=targets,
                                                       train_size=GLOBALS.CONFIG["train_split_percentage"])
        _, train_counts = np.unique(targets[train_indices], return_counts=True)
        _, test_counts = np.unique(targets[test_indices], return_counts=True)
        print(f"Train Class Distribution: {train_counts}")
        print(f"Test Class Distribution: {test_counts}")

        train_dataset = Subset(dataset_train, indices=train_indices)
        test_dataset = Subset(dataset_test, indices=test_indices)

        return train_dataset, test_dataset

    # elif GLOBALS.CONFIG["split_type"] == "random":
    #     train_split = int(len(dataset) * GLOBALS.CONFIG["train_split_percentage"])
    #     test_split = len(dataset) - train_split
    #     train_dataset, test_dataset = random_split(dataset, [train_split, test_split])
    #     return train_dataset, test_dataset


def get_class_weights(train_dataset, labels):
    y_train_indices = train_dataset.indices
    y_train = [labels.iloc[[i]]["Impression"].iloc[0] for i in y_train_indices]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)

    #pos_weight = sum(y_train)/len(y_train)
    pos_weight = (np.array(y_train)==0.).sum()/np.array(y_train).sum()
    #breakpoint()
    #breakpoint()
    return pos_weight


def get_sampler(train_dataset, labels):
    """
    Description: This method defines the sampler to be used by the train loader

    :param train_dataset: The training dataset
    :param labels: The labels for the dataset
    :return: The sampler to be used by the train loader
    """
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
