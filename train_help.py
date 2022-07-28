import os
import sys
import yaml
import torch
import time
import torch.backends.cudnn as cudnn
import global_vars as GLOBALS
import numpy as np
import matplotlib.pyplot as plt

from argparse import Namespace as APNamespace, ArgumentParser
from pathlib import Path

from data import load_data


def load_conf(conf_path):
    with conf_path.open() as f:
        try:
            GLOBALS.CONFIG = yaml.safe_load(f)
            return GLOBALS.CONFIG
        except yaml.YAMLError as exc:
            print(exc)
            print("Could not read from config.yaml, exiting...")
            sys.exit(1)


def get_args(parser: ArgumentParser):
    parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project: Default = '.'")
    parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    parser.add_argument(
        '--output', dest='output',
        default='train_output', type=str,
        help="Set output directory path: Default = 'train_output'")
    parser.add_argument(
        '--dicom_path', dest='dicom_path',
        default='/mnt/8tb_hdd/dicom_store/Myocardial Perfusion Data', type=str,
        help=f"Set dicom directory path: Default = '/mnt/8tb_hdd/dicom_store/Myocardial Perfusion Data/'")
    parser.add_argument(
        '--labels_path', dest='labels_path',
        default='labels.xlsx', type=str,
        help=f"Set labels file path: Default = 'labels.xlsx'")
    parser.add_argument(
        '--station_data_path', dest='station_data_path',
        default='station_data', type=str,
        help=f"Set station data directory path: Default = 'station_data'")


def build_paths(args: APNamespace):
    root_path = Path(args.root).expanduser()
    conf_path = Path(args.config).expanduser()
    out_path = root_path / Path(args.output).expanduser()
    station_data_path = root_path / Path(args.station_data_path).expanduser()

    if not conf_path.exists():
        raise ValueError(f"Config path {conf_path} does not exist")
    if not out_path.exists():
        print(f"Output path {out_path} does not exist, building")
        out_path.mkdir(exist_ok=True, parents=True)
    if not station_data_path.exists():
        print(f"Station path {station_data_path} does not exist, building")
        station_data_path.mkdir(exist_ok=True, parents=True)

    load_conf(conf_path)


def initialize(args: APNamespace, network):
    root_path = Path(args.root).expanduser()
    dicom_path = Path(args.dicom_path).expanduser()
    labels_path = Path(args.labels_path).expanduser()
    station_data_path = root_path / Path(args.station_data_path).expanduser()

    print(f'GLOBALS.CONFIG: {GLOBALS.CONFIG}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pytorch device is set to {device}")
    model = network.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizers = {"SGD": torch.optim.SGD,
                  "Adam": torch.optim.Adam}
    optimizer = optimizers[GLOBALS.CONFIG["optimizer"]](model.parameters(), lr=GLOBALS.CONFIG["learning_rate"])

    loss_functions = {"cross_entropy": torch.nn.CrossEntropyLoss()}
    loss_function = loss_functions[GLOBALS.CONFIG["loss_function"]]

    train_loader, test_loader = load_data(dicom_path, labels_path, station_data_path, GLOBALS.CONFIG["station_info"])

    return model, optimizer, loss_function, train_loader, test_loader


def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / predictions.shape[0]


def evaluate(test_loader, model, loss_function):
    with torch.no_grad():
        running_loss = 0.0
        running_accuracy = 0.0
        for sample in test_loader:
            model.eval()
            images = sample["image"]
            patient_age = sample["patient_age"]
            patient_sex = sample["patient_sex"]
            labels = sample["impression"]

            if torch.cuda.is_available():
                images = images.to(device="cuda", dtype=torch.float)
                patient_age = patient_age.cuda()
                patient_sex = patient_sex.cuda()
                labels = labels.cuda()


            images = torch.unsqueeze(images, dim=1)
            #outputs = model(images, (patient_sex.float(), patient_age.float()))
            outputs = model(images)
            loss = loss_function(outputs, labels)

            predictions = torch.argmax(outputs, 1)
            accuracy = get_accuracy(predictions.cpu().numpy(), labels.cpu().numpy())

            #print(loss,accuracy)

            running_loss += loss.item() * images.shape[0]
            running_accuracy += accuracy * images.shape[0]

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = running_accuracy / len(test_loader.dataset)

        return epoch_loss, epoch_accuracy


def train(model, optimizer, loss_function, train_loader, test_loader, num_epochs, out_path):
    train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = [], [], [], []
    train_stats = {}
    for epoch in range(0, num_epochs, 1):

        time_start = time.time()

        running_loss = 0.0
        running_accuracy = 0.0

        for sample in train_loader:
            #breakpoint()
            model.train()
            images = sample["image"]
            patient_age = sample["patient_age"]
            patient_sex = sample["patient_sex"]
            labels = sample["impression"]
            #breakpoint()

            if torch.cuda.is_available():
                images = images.to(device="cuda", dtype=torch.float)
                patient_age = patient_age.cuda()
                patient_sex = patient_sex.cuda()
                labels = labels.cuda()

            # Forward propagation
            images = torch.unsqueeze(images, dim=1)
            #outputs = model(images, (patient_sex.float(), patient_age.float()))
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predictions = torch.argmax(outputs, 1)
            accuracy = get_accuracy(predictions.cpu().numpy(), labels.cpu().numpy())

            running_loss += loss.item() * images.shape[0]
            running_accuracy += accuracy * images.shape[0]

            #breakpoint()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = running_accuracy / len(train_loader.dataset)
        epoch_test_loss, epoch_test_accuracy = evaluate(test_loader, model, loss_function)

        train_loss_list.append(epoch_train_loss)
        train_accuracy_list.append(epoch_train_accuracy)
        test_loss_list.append(epoch_test_loss)
        test_accuracy_list.append(epoch_test_accuracy)

        time_end = time.time()

        print(
            f"Epoch {epoch + 1}/{num_epochs} Ended | " +
            "Epoch Time: {:.3f}s | ".format(time_end - time_start) +
            "Time Left: {:.3f}s | ".format(
                (time_end - time_start) * (num_epochs - epoch)) +
            "Train Loss: {:.6f} | ".format(epoch_train_loss) +
            "Train Accuracy: {:.6f} | ".format(epoch_train_accuracy) +
            "Test Loss: {:.6f} | ".format(epoch_test_loss) +
            "Test Accuracy: {:.6f}  ".format(epoch_test_accuracy))

        train_stats = {
            "train_loss_list": train_loss_list,
            "train_accuracy_list": train_accuracy_list,
            "test_loss_list": test_loss_list,
            "test_accuracy_list": test_accuracy_list,
            "num_epochs": epoch + 1
        }
        torch.save(train_stats, os.path.join(out_path, 'train_stats.pkl'))

    return train_stats

