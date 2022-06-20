from models.cnn_3d import CNNModel

from train_help import *

if __name__ == "__main__":
    torch.manual_seed(5)
    np.random.seed(5)

    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)

    network = CNNModel()

    model, optimizer, loss_function, train_loader, test_loader = initialize(args, network)

    print('~~Initialization Complete. Beginning training~~')

    train_stats = train(model, optimizer, loss_function, train_loader, test_loader, GLOBALS.CONFIG["num_epochs"])

    print('~~Training Complete. Generating Output Files~~')

    generate_output_files(train_stats, args.output)
