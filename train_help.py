import os
import sys
from typing import Any

import yaml
import torch
import time
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import global_vars as GLOBALS
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from argparse import Namespace as APNamespace, ArgumentParser
from pathlib import Path

from data import load_data

from torchsummary import summary


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

    scheduler = None

    print(f'GLOBALS.CONFIG: {GLOBALS.CONFIG}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pytorch device is set to {device}")
    model = network.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if GLOBALS.CONFIG["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=GLOBALS.CONFIG["learning_rate"],
                                    momentum=GLOBALS.CONFIG["momentum"], weight_decay=GLOBALS.CONFIG["weight_decay"],
                                    dampening=GLOBALS.CONFIG["dampening"])
    elif GLOBALS.CONFIG["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=GLOBALS.CONFIG["learning_rate"])

    if GLOBALS.CONFIG["lr_scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    loss_functions = {"cross_entropy": torch.nn.CrossEntropyLoss()}
    loss_function = loss_functions[GLOBALS.CONFIG["loss_function"]]

    train_loader, test_loader = load_data(dicom_path, labels_path, station_data_path, GLOBALS.CONFIG["station_info"])

    # print(summary(model, [(1, 66, 82, 66), (1, 1)]))
    # breakpoint()

    return model, optimizer, scheduler, loss_function, train_loader, test_loader


def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / predictions.shape[0]


def evaluate(test_loader, model, loss_function):
    predictions_list, labels_list = [], []
    with torch.no_grad():
        running_loss = 0.0
        running_accuracy = 0.0
        for sample in test_loader:
            model.eval()
            images = sample["image"]
            image_list = sample["image_list"]
            patient_age = sample["patient_age"]
            patient_sex = sample["patient_sex"]
            labels = sample["impression"]

            if torch.cuda.is_available():
                images = images.to(device="cuda", dtype=torch.float)
                for i in range(0, len(image_list)):
                    image_list[i] = image_list[i].to(device="cuda", dtype=torch.float32)
                    image_list[i] = torch.unsqueeze(image_list[i], dim=1)
                patient_age = patient_age.cuda()
                patient_sex = patient_sex.cuda()
                labels = labels.cuda()

            images = torch.unsqueeze(images, dim=1)

            outputs = model(images, (patient_sex.float(), patient_age.float()))

            # outputs = model(images, image_list=image_list)
            # outputs = model(images)
            # loss = loss_function(outputs, labels)

            # outputs, output2 = model(image_list[0], image_list[1])
            # outputs = model(image_list[0], image_list[1])
            loss = loss_function(outputs, labels)

            predictions = torch.argmax(outputs, 1)
            accuracy = get_accuracy(predictions.cpu().numpy(), labels.cpu().numpy())

            # print(loss,accuracy)

            running_loss += loss.item() * images.shape[0]
            running_accuracy += accuracy * images.shape[0]

            predictions_list = np.concatenate((predictions_list, predictions.cpu().numpy()))
            labels_list = np.concatenate((labels_list, labels.cpu().numpy()))

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = running_accuracy / len(test_loader.dataset)

        return epoch_loss, epoch_accuracy, predictions_list, labels_list


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


def train(model, optimizer, loss_function, train_loader, test_loader, num_epochs, out_path, scheduler=None):
    train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = [], [], [], []
    train_stats = {}
    for epoch in range(0, num_epochs, 1):

        time_start = time.time()

        running_loss = 0.0
        running_accuracy = 0.0

        for sample in train_loader:
            # breakpoint()
            model.train()
            images = sample["image"]
            image_list = sample["image_list"]
            patient_age = sample["patient_age"]
            patient_sex = sample["patient_sex"]
            labels = sample["impression"]

            # print(sample["image"].shape)
            # breakpoint()
            # import matplotlib.pyplot as plt
            # plt.imshow(images[0, 5, :, :], cmap='gray')
            # plt.show()
            # breakpoint()

            if torch.cuda.is_available():
                images = images.to(device="cuda", dtype=torch.float)
                for i in range(0, len(image_list)):
                    image_list[i] = image_list[i].to(device="cuda", dtype=torch.float32)
                    image_list[i] = torch.unsqueeze(image_list[i], dim=1)
                patient_age = patient_age.cuda()
                patient_sex = patient_sex.cuda()
                labels = labels.cuda()

            # Forward propagation
            images = torch.unsqueeze(images, dim=1)

            outputs = model(images, (patient_sex.float(), patient_age.float()))

            # outputs = model(images, image_list=image_list)
            # outputs = model(images)
            # loss = loss_function(outputs, labels)

            # output1, output2 = model(image_list[0], image_list[1])
            # loss = criterion(output1, output2, labels)
            # breakpoint()

            #outputs = model(image_list[0], image_list[1])
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predictions = torch.argmax(outputs, 1)
            accuracy = get_accuracy(predictions.cpu().numpy(), labels.cpu().numpy())

            running_loss += loss.item() * images.shape[0]
            running_accuracy += accuracy * images.shape[0]

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = running_accuracy / len(train_loader.dataset)
        epoch_test_loss, epoch_test_accuracy, predictions_list, labels_list = evaluate(test_loader, model, loss_function)

        train_loss_list.append(epoch_train_loss)
        train_accuracy_list.append(epoch_train_accuracy)
        test_loss_list.append(epoch_test_loss)
        test_accuracy_list.append(epoch_test_accuracy)

        if scheduler:
            scheduler.step(epoch_test_loss)

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
            "num_epochs": epoch + 1,
            "predictions_list": predictions_list,
            "labels_list": labels_list
        }
        torch.save(train_stats, os.path.join(out_path, 'train_stats.pkl'))

    return train_stats
