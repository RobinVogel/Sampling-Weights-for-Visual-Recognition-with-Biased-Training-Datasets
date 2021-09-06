"""
We compute the distance between the optimal weights
and the best weights.
"""

import json
import click
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import CIFAR10, CIFAR100

from solve_weights import solve_Ws, get_weights
from gen_biased_datasets import get_datasets


def get_res(X, y, K=4, gamma=0.1, seed=42):
    all_classes = np.unique(y)
    datasets, _ = get_datasets(X, y, K=K, gamma=gamma, seed=seed)

    source_datasets = np.array([])
    n_dataset = 0
    omegas = np.empty((0, K))
    all_ys = np.empty(0)
    for _, y, cur_omeg in datasets:
        n_samples = cur_omeg.shape[0]
        cur_source_dataset = (np.ones(n_samples) * n_dataset).astype(int)
        source_datasets = np.concatenate([source_datasets, cur_source_dataset])
        omegas = np.concatenate([omegas, cur_omeg])
        all_ys = np.concatenate([all_ys, y])
        n_dataset += 1

    W = solve_Ws(
        omegas, source_datasets, n_iter=500, batch_size=100, lr=0.001, verbose=False
    )

    inverse_prop = {cy: 1 / np.sum(all_ys == cy) for cy in all_classes}

    weights = get_weights(omegas, source_datasets, W).ravel()
    true_weights = np.array([inverse_prop[cy] for cy in all_ys])
    true_weights /= np.sum(true_weights)

    return np.linalg.norm(weights - true_weights), weights


def plot_res():
    with open("res_norms.json", "rt") as f:
        res_norms = json.load(f)

    plt.figure(figsize=(5, 3))
    indexes = sorted(list(res_norms.keys()))
    data = [res_norms[ind] for ind in indexes]
    indexes = [float(a) for a in indexes]
    widths = list(map(lambda x: 10 ** (-6), np.logspace(-6, -0.1, 12)))
    plt.boxplot(data, positions=indexes, widths=widths)
    # 0.01

    plt.xlabel("$\gamma$")
    # plt.xlim([-0.01, 0.17])
    plt.ylabel("$L_2$ distance from uniform weights")
    plt.tight_layout()
    plt.xscale("log")
    plt.grid()
    plt.savefig("figures/l2_norm_expe.pdf")


def main(name_db, K=3, n_samples=20, seed=42):
    if name_db == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif name_db == "cifar100":
        torch_data = CIFAR100("./data", download=True)

    X = torch_data.data
    y = torch_data.targets
    res = dict()
    res_weights = dict()
    for gamma in np.logspace(-6, -0.1, 12):
        res[gamma] = []
        res_weights[gamma] = []
        for _ in range(0, n_samples):
            norm, weights = get_res(X, y, K=K, gamma=gamma, seed=seed)
            res[gamma].append(norm)
            #  res_weights[gamma].append(weights)
        print("Done Kappa = {:.2f}".format(gamma))

    with open("res_norms.json", "wt") as f:
        json.dump(res, f)

    # with open("res_weights.json", "wt") as f:
    #     json.dump(res_weights, f)


if __name__ == "__main__":
    # main("cifar100", K=5, n_samples=40)
    plot_res()
