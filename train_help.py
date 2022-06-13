import sys
import yaml
import torch
import time
import torch.backends.cudnn as cudnn
import global_vars as GLOBALS

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
    # print("\n---------------------------------")
    # print("MPI-Sunnybrook Train Args")
    # print("---------------------------------\n")
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

    optimizers = {"SGD": torch.optim.SGD}
    optimizer = optimizers[GLOBALS.CONFIG["optimizer"]](model.parameters(), lr=GLOBALS.CONFIG["learning_rate"])

    loss_functions = {"cross_entropy": torch.nn.CrossEntropyLoss()}
    loss_function = loss_functions[GLOBALS.CONFIG["loss_function"]]

    train_loader, test_loader = load_data(dicom_path, labels_path, station_data_path, GLOBALS.CONFIG["station_name"])

    return model, optimizer, loss_function, train_loader, test_loader


def train(model, optimizer, loss_function, train_loader, test_loader, num_epochs):
    for epoch in range(0, num_epochs, 1):

        time_start = time.time()

        for sample in train_loader:
            breakpoint()

        # for i_batch, data in enumerate(train_loader):
        #
        #     breakpoint()
        #     inputs, labels = data
        #     if torch.cuda.is_available():
        #         inputs = inputs.cuda()
        #         labels = labels.cuda()
        #     # Clear gradients
        #     optimizer.zero_grad()
        #     # Forward propagation
        #     outputs = model(inputs)
        #     # Compute loss
        #     loss = loss_function(outputs, labels)
        #     # Compute Gradients and Step
        #     loss.backward()
        #     # Update parameters
        #     optimizer.step()
        #
        #     _, predictions = torch.max(outputs, 1)

        time_end = time.time()

        print(
            f"Epoch {epoch + 1}/{num_epochs} Ended | " +
            "Epoch Time: {:.3f}s | ".format(time_end - time_start) +
            "Time Left: {:.3f}s | ".format(
                (time_end - time_start) * (num_epochs - epoch)) +
            "Train Loss: {:.3f}s ".format(loss.item()))
