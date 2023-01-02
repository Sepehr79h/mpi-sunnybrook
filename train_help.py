import os
import sys
import yaml
import torch
import time
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import global_vars as GLOBALS
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from argparse import Namespace as APNamespace, ArgumentParser
from pathlib import Path
from data import load_data
from sklearn.metrics import f1_score
from torchsummary import summary


def load_conf(conf_path):
    """
    Description: This method loads the config file

    :param conf_path: Path to the config file
    :return: The loaded yaml object containing all config specifications
    """
    with conf_path.open() as f:
        try:
            GLOBALS.CONFIG = yaml.safe_load(f)
            return GLOBALS.CONFIG
        except yaml.YAMLError as exc:
            print(exc)
            print("Could not read from config.yaml, exiting...")
            sys.exit(1)


def get_args(parser: ArgumentParser):
    """
    Description: This method defines all arguments which may be specified

    :param parser: Parser containing all arguments
    """
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
    """
    Description: This method builds all paths specified by the args and loads the config

    :param args: All args specified to the program
    """
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
    """
    Description: This method initializes all necessary parameters to be used in training

    :param args: All args specified to the program
    :param network: The network architecture to be used in training
    :return: model, optimizer, scheduler, loss_function, train_loader, test_loader
    """
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
        optimizer = torch.optim.Adam(model.parameters(), lr=GLOBALS.CONFIG["learning_rate"], weight_decay=GLOBALS.CONFIG["weight_decay"])

    if GLOBALS.CONFIG["lr_scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    loss_functions = {"cross_entropy": torch.nn.CrossEntropyLoss(), "binary_cross_entropy": torch.nn.BCELoss()}
    loss_function = loss_functions[GLOBALS.CONFIG["loss_function"]]

    train_loader, test_loader, weights = load_data(dicom_path, labels_path, station_data_path, GLOBALS.CONFIG["station_info"])

    return model, optimizer, scheduler, loss_function, train_loader, test_loader, weights


def get_accuracy(predictions, labels):
    """
    Description: This method returns the accuracy of the predictions made by the model

    :param predictions: The predictions made by the model
    :param labels: All labels for the dataset
    :return: The accuracy of the predictions made by the model
    """
    return np.sum(predictions == labels) / predictions.shape[0]


def evaluate(test_loader, model, loss_function):
    """
    Description: This method evaluates the model on test data

    :param test_loader: The test loader containing all test data
    :param model: The machine learning model to be used
    :param loss_function: The specified loss function
    :return: epoch_loss, epoch_accuracy, predictions_list, labels_list
    """
    predictions_list, labels_list = [], []
    with torch.no_grad():
        running_loss = 0.0
        running_accuracy = 0.0
        for sample in test_loader:
            model.eval()
            images = sample["image"]
            stress_image = sample["stress_image"]
            rest_image = sample["rest_image"]
            stat_features = sample["stat_features"]
            labels = sample["impression"]
            image_list = sample["image_list"]

            if torch.cuda.is_available():
                for i in range(0, len(image_list)):
                    image_list[i] = image_list[i].to(device="cuda", dtype=torch.float32)
                    image_list[i] = torch.unsqueeze(image_list[i], dim=1)
                images = images.to(device="cuda", dtype=torch.float)
                stress_image = stress_image.to(device="cuda", dtype=torch.float)
                rest_image = rest_image.to(device="cuda", dtype=torch.float)
                stat_features = stat_features.to(device="cuda", dtype=torch.float)
                labels = labels.cuda()

            images = torch.unsqueeze(images, dim=1)
            stress_image = torch.unsqueeze(stress_image, dim=1)
            rest_image = torch.unsqueeze(rest_image, dim=1)

            #breakpoint()
            if GLOBALS.CONFIG["model"] == "BaselineModel":
                outputs = model(stress_image, rest_image, stat_features)
                # outputs = model(stress_image, stat_features)
            elif GLOBALS.CONFIG["model"] == "ResNet3D":
                outputs = model(images, stat_features)
            else:
                outputs = model(images)

            loss = loss_function(outputs.squeeze(), labels.float())
            #loss = loss_function(outputs, labels)
            #breakpoint()
            predictions = torch.where(outputs > 0.5, 1, 0).squeeze()
            #predictions = torch.argmax(outputs, 1)
            #breakpoint()
            accuracy = get_accuracy(predictions.cpu().numpy(), labels.cpu().numpy())

            # print(loss,accuracy)

            running_loss += loss.item() * images.shape[0]
            running_accuracy += accuracy * images.shape[0]

            predictions_list = np.concatenate((predictions_list, predictions.cpu().numpy()))
            labels_list = np.concatenate((labels_list, labels.cpu().numpy()))

        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = running_accuracy / len(test_loader.dataset)

        return epoch_loss, epoch_accuracy, predictions_list, labels_list


def train(model, optimizer, loss_function, train_loader, test_loader, num_epochs, out_path, scheduler=None, weights=None):
    """
    Description: This method performs the training loop and saves all relevant training stats

    :param weights: Class weights for BCE Loss
    :param model: The machine learning model to be used
    :param optimizer: The optimizer to be used
    :param loss_function: The loss function to be used
    :param train_loader: The train loader containing all train data
    :param test_loader: The test loader containing all test data
    :param num_epochs: The number of epochs for the training loop
    :param out_path: The output path to save all training stats
    :param scheduler: The scheduler to be used
    :return: Training stats obtained by the training loop
    """
    #loss_function = torch.nn.CrossEntropyLoss()
    #breakpoint()
    #loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights).cuda())
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
            stress_image = sample["stress_image"]
            rest_image = sample["rest_image"]
            stat_features = sample["stat_features"]
            labels = sample["impression"]
            image_list = sample["image_list"]

            # import matplotlib.pyplot as plt
            # for j in range(0, sample["image"].shape[1]):
            #     plt.subplot(10, 10, j + 1)
            #     plt.imshow(sample["image"][0, j, :, :], cmap='gray')
            # plt.show()
            # breakpoint()

            if torch.cuda.is_available():
                for i in range(0, len(image_list)):
                    image_list[i] = image_list[i].to(device="cuda", dtype=torch.float32)
                    image_list[i] = torch.unsqueeze(image_list[i], dim=1)
                images = images.to(device="cuda", dtype=torch.float)
                stress_image = stress_image.to(device="cuda", dtype=torch.float)
                rest_image = rest_image.to(device="cuda", dtype=torch.float)
                stat_features = stat_features.to(device="cuda", dtype=torch.float)
                labels = labels.cuda()

            #breakpoint()
            # Forward propagation
            images = torch.unsqueeze(images, dim=1)
            stress_image = torch.unsqueeze(stress_image, dim=1)
            rest_image = torch.unsqueeze(rest_image, dim=1)

            #breakpoint()
            if GLOBALS.CONFIG["model"] == "BaselineModel":
                outputs = model(stress_image, rest_image, stat_features)
                #outputs = model(stress_image, stat_features)
            elif GLOBALS.CONFIG["model"] == "ResNet3D":
                outputs = model(images, stat_features)
            else:
                outputs = model(images)
            #breakpoint()
            #breakpoint()
            #breakpoint()
            loss = loss_function(outputs.squeeze(), labels.float())
            #loss = loss_function(outputs, labels)
            #breakpoint()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predictions = torch.where(outputs > 0.5, 1, 0).squeeze()
            #predictions = torch.argmax(outputs, 1)
            accuracy = get_accuracy(predictions.cpu().numpy(), labels.cpu().numpy())

            running_loss += loss.item() * images.shape[0]
            running_accuracy += accuracy * images.shape[0]

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = running_accuracy / len(train_loader.dataset)
        epoch_test_loss, epoch_test_accuracy, predictions_list, labels_list = evaluate(test_loader, model,
                                                                                       loss_function)

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
            "Train Acc: {:.6f} | ".format(epoch_train_accuracy) +
            "Test Loss: {:.6f} | ".format(epoch_test_loss) +
            "Test Acc: {:.6f} | ".format(epoch_test_accuracy) +
            "Balanced Test Acc: {:.6f} | ".format(balanced_accuracy_score(labels_list, predictions_list)) +
            "F1 Score: {:.6f}  ".format(f1_score(labels_list, predictions_list)))

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
