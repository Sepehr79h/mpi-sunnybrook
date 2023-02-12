# from models.cnn_3d import CNNModel
# from models.own_network import CSANet
# from models.lstm import LSTM
# from models.resnet_3d import generate_model
# # from models.densenet_3d import generate_model
# from models.siamese import SiameseNetwork
# from models.s3d import S3D
# from models.i3d import I3D
# # from models.wide_resnet import generate_model
# # from models.resnext import generate_model
# from models.test import Model
from torch.nn import Parameter

from data_analysis.generate_output import generate_output_files
from models import *
from train_help import *


def load_my_state_dict(own_state, loaded_state):
    for name, param in loaded_state.items():
        if "resnet" not in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

if __name__ == "__main__":
    torch.manual_seed(77)
    np.random.seed(77)

    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()
    output_path = build_paths(args)

    network = NetworkRetriever.get_network(GLOBALS.CONFIG["model"])

    print('~~Beginning initialization~~')

    model, optimizer, scheduler, loss_function, train_loader, test_loader, weights = initialize(args, network)

    print('~~Initialization Complete. Beginning training~~')

    if GLOBALS.CONFIG["model"] == "TransferModel":
        load_my_state_dict(model.state_dict(), torch.load('train_output/model.pth'))
        # model.load_state_dict(torch.load('train_output/model.pth'))
        # model.module.fc1 = torch.nn.Linear(in_features=1024, out_features=128, bias=True)
        for name, param in model.named_parameters():
            if "resnet" in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=GLOBALS.CONFIG["learning_rate"], weight_decay=GLOBALS.CONFIG["weight_decay"])
        model = network.to('cuda')
        model = torch.nn.DataParallel(model)

    train_stats, best_model = train(model, optimizer, loss_function, train_loader, test_loader,
                                    GLOBALS.CONFIG["num_epochs"],
                                    args.output, scheduler, weights)

    #print('~~Training Complete. Generating Output Files~~')
    #generate_output_files(train_stats, best_model, args.output)

    print('Finished.')
