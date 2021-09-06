import os
import itertools
from collections import Counter
import numpy as np
import click

import matplotlib

from torchvision.datasets import CIFAR10, CIFAR100

from solve_weights import solve_Ws, get_weights


@click.command()
@click.option("--db_name", default="cifar10", help="Name of the original db.")
@click.option(
    "--hsv_colors/--no_hsv_colors",
    default=False,
    help="Transform the colors to hsv when dividing the data.",
)
@click.option("--border_size", default=2, help="Size of the border.")
@click.option(
    "--in_sample",
    default="sample_inds/balanced.txt",
    help="File where the sample generated is saved.",
)
@click.option(
    "--out_weights",
    default="weights/balanced.txt",
    help="File where the weights for the data are saved.",
)
@click.option(
    "--out_concat_weights",
    default=None,
    help="File where the concatenation weights for the data are saved.",
)
@click.option("--n_iter", default=4000, help="Number of iterations.")
@click.option("--batch_size", default=100, help="Batch size.")
@click.option("--learning_rate", default=0.001, help="Learning rate.")
@click.option(
    "--long_tail_reweight/--no_long_tail_reweight",
    default=False,
    help="Do long tail reweighting.",
)
def main(
    db_name,
    hsv_colors,
    border_size,
    in_sample,
    out_weights,
    out_concat_weights,
    n_iter,
    batch_size,
    learning_rate,
    long_tail_reweight,
):
    if db_name == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif db_name == "cifar100":
        torch_data = CIFAR100("./data", download=True)
    orig_X = torch_data.data
    # y = torch_data.targets
    # n = len(y)
    # n_classes = len(np.unique(y))

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

    # Open the sample
    with open(in_sample, "rt") as f:
        all_samples = [
            [int(ind) for ind in line.split(" ")]
            for line in f.read().strip().split("\n")
        ]
    flat_inds = list()
    for cur_sample in all_samples:
        flat_inds += cur_sample

    if out_concat_weights is not None:
        vec_concat = np.zeros(X.shape[0])
        tot_concat = 0
        for ind in flat_inds:
            vec_concat[ind] += 1
            tot_concat += 1

        res_dict = dict()
        for i, v in enumerate(vec_concat):
            if v > 0:
                res_dict[i] = v / tot_concat

        with open(out_concat_weights, "wt") as f:
            for key, val in res_dict.items():
                f.write("{} {}\n".format(int(key), val))

    # Define the omegas
    mins, maxs = list(), list()
    for sample in all_samples:
        mins.append(np.min(avg_borders[sample], axis=0) - np.finfo(float).eps)
        maxs.append(np.max(avg_borders[sample], axis=0) + np.finfo(float).eps)

    mins, maxs = np.array(mins), np.array(maxs)
    # print(mins)
    # print(maxs)
    X_sample = avg_borders[sample]

    if long_tail_reweight:
        with open("obs_per_bin/power_law.txt", "rt") as f:
            lg_card = [int(a) for a in f.read().strip().split("\n")]
        with open("emp_obs_per_bin/last_assignment.txt", "rt") as f:
            as_card = [int(a) for a in f.read().strip().split("\n")]

    K = len(all_samples)
    omegas = list()
    for x in X_sample:
        cvals = list()
        for k in range(0, K):
            is_ok = np.all(
                [(v >= mins[k][d] and v <= maxs[k][d]) for d, v in enumerate(x)]
            )

            if is_ok:
                tmp_val = 1
            else:
                tmp_val = 0

            if long_tail_reweight:
                tmp_val *= as_card[k] / lg_card[k]
            cvals.append(tmp_val)

        cvals = np.array(cvals)
        omegas.append(cvals)
    omegas = np.array(omegas)

    # Solve for weights
    source_datasets = list()
    for i, sample in enumerate(all_samples):
        source_datasets += [i] * len(sample)
    source_datasets = np.array(source_datasets)

    W = solve_Ws(
        omegas,
        source_datasets,
        n_iter=n_iter,
        batch_size=batch_size,
        lr=learning_rate,
        verbose=True,
    )

    weights = get_weights(omegas, source_datasets, W).ravel()
    ind_val, ind_count = np.unique(flat_inds, return_counts=True)
    ind_to_count = {val: count for val, count in zip(ind_val, ind_count)}

    # Save the weights
    if not os.path.exists("weights"):
        os.makedirs("weights")

    res_dict = dict()
    for ind, wei in zip(flat_inds, weights):
        if ind in res_dict:
            res_dict[ind] += wei / ind_to_count[ind]
        else:
            res_dict[ind] = wei / ind_to_count[ind]

    with open(out_weights, "wt") as f:
        for key, val in res_dict.items():
            f.write("{} {}\n".format(int(key), val))


if __name__ == "__main__":
    main()
