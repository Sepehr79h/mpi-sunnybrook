import pydicom
import os
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt


def save_station_info(dicom_path='/mnt/8tb_hdd/dicom_store/Myocardial Perfusion Data/'):
    station_info = {}

    time_start = time.time()

    for root, dirs, files in tqdm(os.walk(dicom_path)):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext == ".dcm":
                try:
                    ds = pydicom.dcmread(os.path.join(root, file))
                    if ds.StationName not in station_info.keys():
                        station_info[ds.StationName] = {}
                    if ds.SeriesDescription not in station_info[ds.StationName].keys():
                        station_info[ds.StationName][ds.SeriesDescription] = 0
                    station_info[ds.StationName][ds.SeriesDescription] += 1
                except:
                    print("Cannot read file: ", os.path.join(root, file))

    with open('station_info.json', 'w') as fp:
        json.dump(station_info, fp, indent=4)

    time_end = time.time()

    print("Station info saved. Time taken: {:.2f} s".format(time_end - time_start))


def plot_station_histograms():
    f = open('station_info.json')
    station_info = json.load(f)
    for station in station_info:
        x = list(station_info[station].keys())
        y = list(station_info[station].values())
        fig = plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 2})
        plt.bar(x, y, color='maroon', width=0.4)
        fig.autofmt_xdate()
        plt.savefig(f'plots/{station.replace("/","_")}_histogram.jpg', dpi=400)


if __name__ == "__main__":
    save_station_info()
    #plot_station_histograms()
