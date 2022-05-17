from models.cnn_3d import CNNModel

from train_help import *

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)

    network = CNNModel()

    model, optimizer = initialize(network)

    print('~~Initialization Complete. Beginning training~~')
