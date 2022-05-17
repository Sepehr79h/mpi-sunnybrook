import global_vars as GLOBALS
import sys
import yaml
import torch
from argparse import Namespace as APNamespace, ArgumentParser
from pathlib import Path


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


def load_data():
    # temporary sample data

    batch_size = 2

    train_x = torch.rand(50, 40, 64, 64)  # num images, num channels, img height, img width
    train_y = torch.randint(1, 5, (50, 1))
    train_x = torch.unsqueeze(train_x, dim=1)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader


def initialize(network):
    print(f'GLOBALS.CONFIG: {GLOBALS.CONFIG}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pytorch device is set to {device}")
    model = network.to(device)

    optimizers = {"SGD": torch.optim.SGD}
    optimizer = optimizers[GLOBALS.CONFIG["optimizer"]](model.parameters(), lr=GLOBALS.CONFIG["learning_rate"])

    return model, optimizer
