import click
import torch
import json

import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100

from gen_biased_datasets import get_datasets
from solve_weights import solve_Ws, get_weights


@click.command()
@click.option("--name_db", default="cifar10", help="Name of the original db.")
@click.option("--n_datasets", default=5, help="Number of dbs (K).")
@click.option("--gamma", default=0.1, help="Kappa value for db generation.")
@click.option("--outfile", default="weights/weights.txt", help=".")
@click.option("--n_iter", default=500, help="Number of iterations.")
@click.option("--batch_size", default=100, help="Batch size.")
@click.option("--learning_rate", default=0.01, help="Learning rate.")
def main(
    name_db,
    n_datasets,
    gamma,
    outfile,
    n_iter,
    batch_size,
    learning_rate,
):
    if name_db == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif name_db == "cifar100":
        torch_data = CIFAR100("./data", download=True)
    X = torch_data.data
    y = torch_data.targets

    datasets, all_inds = get_datasets(X, y, K=n_datasets, gamma=gamma, seed=42)

    flat_inds = np.array([])
    source_datasets = np.array([])
    n_dataset = 0
    omegas = np.empty((0, n_datasets))
    for dataset_part, ind_part in zip(datasets, all_inds):
        _, y, cur_omeg = dataset_part

        n_samples = cur_omeg.shape[0]
        cur_source_dataset = (np.ones(n_samples) * n_dataset).astype(int)
        source_datasets = np.concatenate([source_datasets, cur_source_dataset])
        omegas = np.concatenate([omegas, cur_omeg])

        flat_inds = np.concatenate([flat_inds, ind_part])
        n_dataset += 1

    W = solve_Ws(
        omegas,
        source_datasets,
        n_iter=n_iter,
        batch_size=batch_size,
        lr=learning_rate,
        verbose=False,
    )
    weights = get_weights(omegas, source_datasets, W).ravel()
    ind_val, ind_count = np.unique(flat_inds, return_counts=True)
    ind_to_count = {val: count for val, count in zip(ind_val, ind_count)}

    res_dict = dict()
    for ind, wei in zip(flat_inds, weights):
        if ind in res_dict:
            res_dict[ind] += wei / ind_to_count[ind]
        else:
            res_dict[ind] = wei / ind_to_count[ind]

    with open(outfile, "wt") as f:
        for key, val in res_dict.items():
            f.write("{} {}\n".format(int(key), val))


if __name__ == "__main__":
    main()
