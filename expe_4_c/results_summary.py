import os
import json
import glob

import click
import numpy as np
import matplotlib.pyplot as plt


def get_table(db_name="cifar10", debug=False, concat=False, alpha=False):
    gammas = ["0.1", "1", "10"]  # "0.01",
    # ["0.5", "1", "2", "4"]
    means = list()
    stds = list()
    n_measures = list()
    print(" ".join(gammas))
    for gamma in gammas:
        if concat and alpha:
            glob_name = "models/{}_gamma_{}_alpha_0.75_ite[0-9]_concat-*.json".format(
                db_name, gamma
            )
        elif alpha:
            glob_name = "models/{}_gamma_{}_alpha_0.75_ite[0-9]-*.json".format(
                db_name, gamma
            )
        elif concat:
            glob_name = "models/{}_gamma_{}_ite[0-9]_concat-*.json".format(
                db_name, gamma
            )
        else:
            glob_name = "models/{}_gamma_{}_ite[0-9]-*.json".format(db_name, gamma)
        # print(glob_name)
        cand_fnames = glob.glob(glob_name)

        cur_vals = list()
        for cand_fname in cand_fnames:
            with open(cand_fname, "rt") as f:
                d = json.load(f)
                cur_vals.append(d["avg_top1"])
        if len(cur_vals) > 0:
            means.append(np.mean(cur_vals))
            stds.append(np.std(cur_vals))
            n_measures.append(len(cur_vals))
        else:
            means.append(-999)
            stds.append(-999)
            n_measures.append(0)
    print(
        "\n".join(
            [
                "${:.2f} \; (\\pm {:.2f})$".format(100 * mean, 100 * 2 * std)
                for mean, std in zip(means, stds)
            ]
        )
    )
    print(" & ".join(["{} ".format(a) for a in n_measures]))


def main():
    for db_name in ["cifar10", "cifar100"]:
        print("\nWorking with {}".format(db_name))
        get_table(db_name=db_name)
        print("\nWorking with {} - concat".format(db_name))
        get_table(db_name=db_name, concat=True)

        print("\nWorking with {} - ALPHA = 0.75".format(db_name))
        get_table(db_name=db_name, alpha=True)
        # print("Working with {} - CONCAT".format(db_name))
        # get_table(db_name=db_name, concat=True)
        print("\nWorking with {} - ALPHA = 0.75 and concat".format(db_name))
        get_table(db_name=db_name, alpha=True, concat=True)
        print("\n\n\n")


if __name__ == "__main__":
    main()
