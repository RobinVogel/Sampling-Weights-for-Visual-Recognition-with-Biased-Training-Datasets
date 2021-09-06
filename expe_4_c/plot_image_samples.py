import os
import itertools
from collections import Counter

import click
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision.datasets import CIFAR10, CIFAR100


@click.command()
@click.option("--name_db", default="cifar10", help="Name of the original db.")
@click.option(
    "--in_assignment",
    default="sample_inds/balanced.txt",
    help="File where the sample generated is saved.",
)
@click.option(
    "--n_img_per_class",
    default=100,
    help="Number of images to save.",
)
@click.option(
    "--outfolder",
    default="figures/balanced",
    help="Folder where the different pictures will be saved.",
)
def main(name_db, in_assignment, n_img_per_class, outfolder):
    if name_db == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif name_db == "cifar100":
        torch_data = CIFAR100("./data", download=True)
    X = torch_data.data
    y = torch_data.targets

    with open(in_assignment, "rt") as f:
        all_inds = [
            [int(a) for a in l.split(" ")] for l in f.read().strip().split("\n")
        ]

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    n_samples = len(all_inds)
    for i in range(0, n_samples):
        imgs = X[np.random.choice(all_inds[i], n_img_per_class)]
        path_sample = "{}/sample_{}".format(outfolder, i)
        if not os.path.exists(path_sample):
            os.makedirs(path_sample)

        for j, img in enumerate(imgs):
            im = Image.fromarray(img)
            im.save("{}/sample_{}/{}.jpg".format(outfolder, i, j))


if __name__ == "__main__":
    main()
