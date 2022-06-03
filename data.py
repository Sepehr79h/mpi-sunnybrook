import torch
import os
import pydicom
import pickle
import time
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


def load_data(dicom_path, labels_path, station_data_path, station_name):
    print(f"Fetching dicom store from path: {dicom_path}")
    print(f"Fetching labels from path: {labels_path}")

    if not os.path.exists(os.path.join(station_data_path, f"dicom_{station_name}.pkl")):
        # create data object holding all data, save to file for fast loading
        save_dicom_store(dicom_path, station_data_path, station_name)

    with open(os.path.join(station_data_path, f"dicom_{station_name}.pkl"), 'rb') as f:
        data = pickle.load(f)

    print(f"Dicom data loaded successfully from station: {station_name}")

    breakpoint()


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
