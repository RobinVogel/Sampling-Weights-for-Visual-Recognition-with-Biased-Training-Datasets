import sys
import os
import glob
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10, CIFAR100


def transform_data(orig_X, border_size=2, hsv_colors=True):
    if hsv_colors:
        new_X = list()
        for x in orig_X:
            new_X.append(matplotlib.colors.rgb_to_hsv(x))
        new_X = np.array(new_X)
        new_X[:, :, :, 2] = new_X[:, :, :, 2] / 255
        X = new_X
    else:
        X = orig_X

    border1 = X[:, :border_size, :, :]
    border2 = X[:, :, :border_size, :]
    border3 = X[:, (32 - border_size) :, :, :]
    border4 = X[:, :, (32 - border_size) :, :]
    all_borders = np.concatenate(
        [border1, border2.swapaxes(1, 2), border3, border4.swapaxes(1, 2)], axis=1
    )
    avg_borders = np.mean(all_borders, axis=(1, 2))
    return avg_borders


def main(db_name, gamma, outname, cdf=False, hsv_colors=True):
    if db_name == "cifar10":
        cifar_train = CIFAR10("./data", download=True, train=True)
        cifar_val = CIFAR10("./data", download=True, train=False)
        # n_classes = 10
    elif db_name == "cifar100":
        cifar_train = CIFAR100("./data", download=True, train=True)
        cifar_val = CIFAR100("./data", download=True, train=False)
        # n_classes = 100

    X_train = transform_data(cifar_train.data)
    # y_train = np.array(cifar_train.targets)
    X_val = transform_data(cifar_val.data)
    # y_val = np.array(cifar_val.targets)

    glob_name = "weights/{}_gamma_{}_ite*.txt".format(db_name, gamma)

    all_weight_files = glob.glob(glob_name)

    fig, subplots = plt.subplots(1, 3, figsize=(6, 2))
    alpha_val = 0.2

    for coord, type_plot in enumerate(["H", "S", "V"]):
        x = np.sort(X_val[:, 1])
        y = np.arange(len(x)) / float(len(x))
        subplots[coord].plot(x, y, label="val cdf", color="black", alpha=0.8)

        first = True
        for weight_file in all_weight_files:
            xs = list()
            weights = list()
            with open(weight_file, "rt") as f:
                for line in f.read().strip().split("\n"):
                    index, cweight = line.split(" ")
                    index, cweight = int(index), float(cweight)
                    xs.append(X_train[index, coord])
                    weights.append(cweight)
            xs = np.array(xs)
            weights = np.array(weights) / np.sum(weights)
            order = np.argsort(xs)
            # print(xs.max())
            x = xs[order]
            y = weights[order].cumsum()
            if first:
                subplots[coord].plot(
                    x,
                    y,
                    label="train cdf",
                    color="black",
                    linestyle="--",
                    alpha=alpha_val,
                )
                first = False
            else:
                subplots[coord].plot(
                    x, y, color="black", linestyle="--", alpha=alpha_val
                )

        # subplots[coord].xlabel("y")
        # subplots[coord].ylabel("pdf over classes")
        subplots[coord].set_title("Color dim {}".format(type_plot))
        # subplots[coord].legend()
        subplots[coord].grid()

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in subplots.flat:
        ax.label_outer()

    fig.suptitle("$\gamma= {}$".format(gamma))
    plt.tight_layout()
    plt.savefig("{}.pdf".format(outname))
    plt.close()


if __name__ == "__main__":
    debug = False
    if debug:
        main("cifar10", "0.001", "figures/dists/test")
        sys.exit()

    for db_name in ["cifar10", "cifar100"]:
        if not os.path.exists("figures/dists/{}".format(db_name)):
            os.makedirs("figures/dists/{}".format(db_name))

        # "0.001", "0.2", "0.5", "2", "4"
        for gamma in ["0.1", "1", "10"]:
            main(db_name, gamma, "figures/dists/{}_gamma{}.pdf".format(db_name, gamma))
