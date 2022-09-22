import sys

import matplotlib
import seaborn as sn
import pandas as pd

sys.path.insert(0, '.')

from train_help import *
from sklearn.metrics import confusion_matrix

#import global_vars as GLOBALS


def generate_output_files(train_stats, out_path):
    # matplotlib.use('GTK')
    x = np.arange(0, train_stats["num_epochs"])
    plt.plot(x, train_stats["train_loss_list"], label="train")
    plt.plot(x, train_stats["test_loss_list"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{out_path}/loss.jpg', dpi=400)
    # plt.show()
    plt.clf()
    plt.plot(x, train_stats["train_accuracy_list"], label="train")
    plt.plot(x, train_stats["test_accuracy_list"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'{out_path}/accuracy.jpg', dpi=400)
    # plt.show()

    build_confusion_matrix(train_stats, out_path)


def build_confusion_matrix(train_stats, out_path):
    cm = confusion_matrix(train_stats["labels_list"], train_stats["predictions_list"])
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sn.heatmap(cmn, annot=True, fmt='.2f', xticklabels=GLOBALS.CONFIG["classes"], yticklabels=GLOBALS.CONFIG["classes"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{out_path}/confusion_matrix.jpg')


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    get_args(parser)
    args = parser.parse_args()

    stats = torch.load(os.path.join(args.output, 'train_stats.pkl'))
    generate_output_files(stats, args.output)
    #build_confusion_matrix(stats, args.output)
