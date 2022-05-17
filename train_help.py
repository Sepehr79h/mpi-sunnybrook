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
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    parser.add_argument(
        '--output', dest='output',
        default='train_output', type=str,
        help="Set output directory path: Default = 'train_output'")


def build_paths(args: APNamespace):
    conf_path = Path(args.config).expanduser()
    out_path = Path(args.output).expanduser()

    if not conf_path.exists():
        raise ValueError(f"Config path {conf_path} does not exist")
    if not out_path.exists():
        print(f"Output path {out_path} does not exist, building")
        out_path.mkdir(exist_ok=True, parents=True)

    load_conf(conf_path)


def initialize(network):
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

    train_loader, test_loader = load_data()

    return model, optimizer, loss_function, train_loader, test_loader


def train(model, optimizer, loss_function, train_loader, test_loader, num_epochs):
    for epoch in range(0, num_epochs, 1):

        time_start = time.time()

        for i_batch, data in enumerate(train_loader):
            inputs, labels = data
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(inputs)
            # Compute loss
            loss = loss_function(outputs, labels)
            # Compute Gradients and Step
            loss.backward()
            # Update parameters
            optimizer.step()

            _, predictions = torch.max(outputs, 1)

        time_end = time.time()

        print(
            f"Epoch {epoch+1}/{num_epochs} Ended | " +
            "Epoch Time: {:.3f}s | ".format(time_end - time_start) +
            "Time Left: {:.3f}s | ".format(
                (time_end - time_start) * (num_epochs - epoch)) +
            "Train Loss: {:.3f}s ".format(loss.item()))
