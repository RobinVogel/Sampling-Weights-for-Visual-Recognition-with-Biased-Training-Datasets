import os
import itertools
from collections import Counter
import numpy as np
import click

import matplotlib

from torchvision.datasets import CIFAR10, CIFAR100


@click.command()
@click.option("--db_name", default="cifar10", help="Name of the original db.")
@click.option(
    "--hsv_colors/--no_hsv_colors",
    default=False,
    help="Transform the colors to hsv when dividing the data.",
)
@click.option("--border_size", default=2, help="Size of the border.")
@click.option("--gamma", default=0.1, help="Size of the border.")
@click.option(
    "--obs_per_bin",
    default="obs_per_bin/balanced.txt",
    help="File that contains the number of observations per bin.",
)
@click.option(
    "--out_sample",
    default="sample_inds/balanced.txt",
    help="File where the sample generated is saved.",
)
def main(db_name, hsv_colors, border_size, gamma, obs_per_bin, out_sample):
    if db_name == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif db_name == "cifar100":
        torch_data = CIFAR100("./data", download=True)
    orig_X = torch_data.data
    y = torch_data.targets
    n = len(y)
    gamma = float(gamma)
    n_classes = len(np.unique(y))

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
    meds = np.median(avg_borders, axis=0)
    binary_filter = (avg_borders - meds.reshape((1, -1)) >= 0).astype(int)
    assignment = (
        binary_filter.dot(np.array([[4, 2, 1]]).transpose()).astype(int).ravel()
    )
    counter_assign = Counter(assignment)

    if not os.path.exists("emp_obs_per_bin"):
        os.makedirs("emp_obs_per_bin")

    with open("emp_obs_per_bin/last_assignment.txt", "wt") as f:
        for i in range(0, 8):
            f.write(str(counter_assign[i]) + "\n")

    with open("emp_obs_per_bin/last_class_assignment.txt", "wt") as f:
        for i in range(0, 8):
            d = Counter(np.array(y)[assignment == i])
            line = " ".join([str(d.get(cy, 0)) for cy in range(0, n_classes)])
            f.write(line + "\n")

    with open(obs_per_bin, "rt") as f:
        n_obs_to_sample = [int(a) for a in f.read().strip().split("\n")]

    all_values = list(itertools.product(*[(False, True)] * 3))
    all_samples = list()
    for i_dataset in range(0, 8):
        print("On dataset {}".format(i_dataset))
        # Compute the sampling weights
        class_code = np.array(all_values[i_dataset])
        weights = list()
        for i, x in enumerate(avg_borders):
            if (binary_filter[i] == class_code).all():
                weights.append(1)
            else:
                dist = 0
                for d in range(0, 3):
                    if (x[d] - meds[d]) * (2 * class_code[d] - 1) < 0:
                        dist += abs(x[d] - meds[d])
                weights.append(max(0, 1 - dist / gamma))
        weights = np.array(weights)

        # Sample a number of elements equal to the number of elements in the
        # quadrant by default.
        cur_sample = np.random.choice(
            range(0, n),
            size=n_obs_to_sample[i_dataset],
            p=weights / weights.sum(),
        )
        all_samples.append(cur_sample)

    if not os.path.exists("sample_inds"):
        os.makedirs("sample_inds")

    # Save the sample
    with open(out_sample, "wt") as f:
        for sample in all_samples:
            f.write(" ".join([str(a) for a in sample]))
            f.write("\n")


if __name__ == "__main__":
    main()
