from models.cnn_3d import CNNModel
from models.own_network import CSANet
from models.resnet_3d import generate_model
#from models.densenet_3d import generate_model
from models.siamese import SiameseNetwork
from models.s3d import S3D
from models.i3d import I3D
from data_analysis.generate_plots import generate_output_files
#from models.wide_resnet import generate_model
#from models.resnext import generate_model

from train_help import *

if __name__ == "__main__":
    torch.manual_seed(77)
    np.random.seed(77)

    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)

    #network = SiameseNetwork()
    #network = CSANet()
    #network = CNNModel(num_classes=len(GLOBALS.CONFIG["classes"]))
    network = generate_model(18, n_input_channels=1, n_classes=len(GLOBALS.CONFIG["classes"]))
    #network = generate_model(121, n_input_channels=1, num_classes=len(GLOBALS.CONFIG["classes"]))
    #network = S3D(num_class=len(GLOBALS.CONFIG["classes"]))
    #network = I3D(num_classes=2, input_channel=1)
    #network = generate_model(50, 1, n_input_channels=1, n_classes=len(GLOBALS.CONFIG["classes"]))

    model, optimizer, scheduler, loss_function, train_loader, test_loader = initialize(args, network)

    print('~~Initialization Complete. Beginning training~~')

    train_stats = train(model, optimizer, scheduler, loss_function, train_loader, test_loader, GLOBALS.CONFIG["num_epochs"], args.output)

    print('~~Training Complete. Generating Output Files~~')

    generate_output_files(train_stats, args.output)
