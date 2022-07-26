import sys

sys.path.insert(0, '.')

from train_help import *


def generate_output_files(train_stats, out_path):
    x = np.arange(0, train_stats["num_epochs"])
    plt.plot(x, train_stats["train_loss_list"], label="train")
    plt.plot(x, train_stats["test_loss_list"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{out_path}/loss.jpg', dpi=400)
    plt.show()
    plt.clf()
    plt.plot(x, train_stats["train_accuracy_list"], label="train")
    plt.plot(x, train_stats["test_accuracy_list"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'{out_path}/accuracy.jpg', dpi=400)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()

    stats = torch.load(os.path.join(args.output, 'train_stats.pkl'))
    generate_output_files(stats, args.output)
