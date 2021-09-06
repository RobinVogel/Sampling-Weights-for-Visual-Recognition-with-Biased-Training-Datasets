import glob
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10, CIFAR100


def main(db_name, gamma, outname, cdf=False):
    if db_name == "cifar10":
        cifar_train = CIFAR10("./data", download=True, train=True)
        cifar_val = CIFAR10("./data", download=True, train=False)
        n_classes = 10
    elif db_name == "cifar100":
        cifar_train = CIFAR100("./data", download=True, train=True)
        cifar_val = CIFAR100("./data", download=True, train=False)
        n_classes = 100

    # X_train = cifar_train.data
    y_train = np.array(cifar_train.targets)
    # X_val = cifar_val.data
    y_val = np.array(cifar_val.targets)

    dist_val = np.zeros(n_classes)
    for y in y_val:
        dist_val[y] += 1
    dist_val /= y_val.shape[0]

    plt.figure(figsize=(5, 3))
    if cdf:
        dist_val = dist_val.cumsum()
    plt.plot(
        range(0, n_classes),
        dist_val,
        label="val cdf" if cdf else "val pdf",
        color="black",
    )

    glob_name = "weights/{}_gamma{}_ite*.txt".format(db_name, gamma)
    all_weight_files = glob.glob(glob_name)

    first = True

    for weight_file in all_weight_files:
        index_to_weight = dict()
        with open(weight_file, "rt") as f:
            for line in f.read().strip().split("\n"):
                index, cweight = line.split(" ")
                index, cweight = int(index), float(cweight)
                index_to_weight[index] = cweight

        dist_train = np.zeros(n_classes)
        tot_weight = 0
        for index, weight in index_to_weight.items():
            dist_train[y_train[index]] += weight
            tot_weight += weight
        dist_train /= tot_weight
        if cdf:
            dist_train = dist_train.cumsum()

        if first:
            plt.plot(
                range(0, n_classes),
                dist_train,
                label="train cdf" if cdf else "train pdf",
                color="black",
                linestyle="--",
                alpha=0.8,
            )
            first = False
        else:
            plt.plot(
                range(0, n_classes),
                dist_train,
                color="black",
                linestyle="--",
                alpha=0.8,
            )
    plt.xlabel("y")
    plt.title("$\gamma = {}$".format(gamma))
    plt.ylabel("cdf over classes" if cdf else "pdf over classes")
    if db_name == "cifar10":
        plt.ylim([0.0, 0.30])
    elif db_name == "cifar100":
        plt.ylim([0.0, 0.03])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(outname)


if __name__ == "__main__":
    for db_name in ["cifar10", "cifar100"]:
        for gamma in ["0.001", "0.01", "0.1", "0.2"]:
            main(db_name, gamma, "figures/dists/{}_gamma{}.pdf".format(db_name, gamma))
