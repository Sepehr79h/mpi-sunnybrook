from models.cnn_3d import CNNModel
from models.resnet_3d import generate_model
from data_analysis.generate_plots import generate_output_files

from train_help import *

if __name__ == "__main__":
    torch.manual_seed(77)
    np.random.seed(77)

    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)

    #network = CNNModel()
    network = generate_model(10, n_input_channels=1, n_classes=4)

    model, optimizer, loss_function, train_loader, test_loader = initialize(args, network)

    print('~~Initialization Complete. Beginning training~~')

    train_stats = train(model, optimizer, loss_function, train_loader, test_loader, GLOBALS.CONFIG["num_epochs"], args.output)

    print('~~Training Complete. Generating Output Files~~')

    generate_output_files(train_stats, args.output)
