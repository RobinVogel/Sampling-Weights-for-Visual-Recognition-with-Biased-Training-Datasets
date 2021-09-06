import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
from collections import Counter
from torchvision.datasets import CIFAR10, CIFAR100


def get_datasets(X, y, K=3, gamma=0.1, size_tot=50000, seed=42, permute=False):
    np.random.seed(seed)
    y = np.array(y)
    classes = np.unique(y)
    n_classes = len(classes)
    n_c = int(n_classes / K)

    ind_per_class = dict()
    for c_class in classes:
        ind_per_class[c_class] = np.where(y == c_class)[0]

    if permute:
        perm_class = np.random.permutation(classes)
        map_class = {classes[i]: perm_class[i] for i, _ in enumerate(perm_class)}
        for c_class in classes:
            ind_per_class[map_class[c_class]] = np.where(y == c_class)[0]

    alpha = 1 / K
    size_per_db = [int(alpha * size_tot)] * K

    split_datasets = list()
    all_inds = list()
    n_per_class_by_db = list()

    for k in range(0, K):
        # Define the pk's
        pk = np.zeros(n_classes)

        for i in range(k * n_c, (k + 1) * n_c):
            pk[i % n_classes] = (1 - gamma) / n_c
        for i in range((k + 1) * n_c, (k + 2) * n_c):
            pk[i % n_classes] = gamma / n_c

        n_per_class_by_db.append(np.random.multinomial(size_per_db[k], pk))

    for k in range(0, K):
        # Do the generation by random sampling
        n_per_class = n_per_class_by_db[k]
        cur_inds = np.concatenate(
            [
                np.random.choice(ind_per_class[c_cl], size=n_cl)
                for c_cl, n_cl in zip(classes, n_per_class)
            ]
        )
        cur_X, cur_y = X[cur_inds], y[cur_inds]
        cur_omeg = np.array(
            [[n_per_class_by_db[k][y] for k in range(0, K)] for y in cur_y]
        )

        # Append them
        split_datasets.append((cur_X, cur_y, cur_omeg))
        all_inds.append(cur_inds)

    return split_datasets, all_inds


def stacking_histogram_datasets(name_db, K=5, gamma=0.1, seed=42):
    # plt.style.use("ggplot")
    # hsv_colors = [[((72 * i) % 360) / 360, 100 / 100, 65 / 100] for i in range(0, 24)]
    # rgb_colors = diverging_hcl(h=[0, 360], c=100, l=65)(5)
    # print(rgb_colors)
    colors = iter([plt.cm.tab10(i) for i in range(20)])
    colors = iter(
        [
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            "#e41a1c",
            "#dede00",
        ]
    )

    if name_db == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif name_db == "cifar100":
        torch_data = CIFAR100("./data", download=True)

    X = torch_data.data
    y = torch_data.targets

    datasets, _ = get_datasets(X, y, K=K, gamma=gamma, seed=seed)

    plt.figure(figsize=(4, 4))

    Ys = [ds[1].astype(int) for ds in datasets]
    all_ys = np.unique(np.concatenate(Ys))
    bottom = np.zeros(len(all_ys))
    for i, y in enumerate(Ys):
        labels, values = list(), list()
        count = Counter(y)
        print("dataset", i, count)
        for lab, val in count.items():
            labels.append(lab)
            values.append(val)
        plt.bar(
            labels,
            values,
            width=0.5,
            bottom=[bottom[a] for a in labels],
            label="$D_{}$".format(i),
            color=next(colors),
        )
        for lab, val in zip(labels, values):
            bottom[lab] += val

    plt.legend()
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.grid()
    plt.xlabel("$y$")
    plt.ylabel("# obs")
    plt.title("$\gamma$ = {:.2f}".format(gamma))
    plt.savefig("figures/{}_gamma{:.2f}.pdf".format(name_db, gamma))


def main():
    for name_db in ["cifar10", "cifar100"]:
        for gamma in [0.05, 0.20]:
            # gamma = 0.01 / 0.2
            stacking_histogram_datasets(name_db, gamma=gamma)


if __name__ == "__main__":
    main()
