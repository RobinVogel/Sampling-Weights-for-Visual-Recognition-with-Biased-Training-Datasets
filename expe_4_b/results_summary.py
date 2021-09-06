import glob
import json
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_table(db_name="cifar10", debug=False):
    gammas = [0.001, 0.01, 0.1, 0.2]
    means = list()
    stds = list()

    glob_name = "models/{}_ite*.json".format(db_name)
    cand_fnames = glob.glob(glob_name)

    cur_vals = list()
    for cand_fname in cand_fnames:
        with open(cand_fname, "rt") as f:
            d = json.load(f)
            cur_vals.append(d["avg_top1"])

    mean = np.mean(cur_vals)
    std = np.std(cur_vals)
    print("{} {:.2f} (\\pm {:.2f})".format(db_name, 100 * mean, 100 * std))

    print(" ".join(["{:.2f}".format(gamma) for gamma in gammas]))
    for gamma in gammas:
        glob_name = "models/{}_gamma{}_*.json".format(db_name, gamma)
        cand_fnames = glob.glob(glob_name)

        cur_vals = list()
        for cand_fname in cand_fnames:
            with open(cand_fname, "rt") as f:
                d = json.load(f)
                cur_vals.append(d["avg_top1"])
        means.append(np.mean(cur_vals))
        stds.append(np.std(cur_vals))
    print(
        " ".join(
            [
                "{:.2f} (\\pm {:.2f})".format(100 * mean, 100 * 2 * std)
                for mean, std in zip(means, stds)
            ]
        )
    )


def plot_results(db_name="cifar10", debug=False):
    gammas = [0.001, 0.01, 0.1, 0.2]
    no_ites = range(0, 8)

    perfos = {"gamma": [], "iter": [], "acc": []}
    for gamma in gammas:
        for no_ite in no_ites:
            glob_name = "models/{}_gamma{}_ite{}-*.json".format(db_name, gamma, no_ite)
            cand_fnames = glob.glob(glob_name)

            for cand_fname in cand_fnames:
                perfos["gamma"].append(gamma)
                perfos["iter"].append(no_ite)
                with open(cand_fname, "rt") as f:
                    d = json.load(f)
                    perfos["acc"].append(d["avg_top1"])

    df = pd.DataFrame(perfos)

    # plot several gamma's on the same graph with noise as x axis
    # and accuracy as y axis

    plt.figure(figsize=(4, 4))

    data = list()
    colors = ["blue", "red", "green", "black"]
    for color, gamma in zip(colors, gammas):
        cur_df = df[df["gamma"] == gamma]
        cur_vals = cur_df["acc"]
        data.append(cur_vals)
    indexes = gammas
    widths = list(map(lambda x: x / 4, gammas))

    plt.boxplot(data, positions=indexes, widths=widths)
    # if db_name == "cifar10":
    #     plt.ylim([0.82, 0.94])
    # elif db_name == "cifar100":
    #     plt.ylim([0.64, 0.70])
    plt.grid()
    plt.xlabel("gamma")
    plt.xscale("log")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/boxplot_{}.pdf".format(db_name))


@click.command()
@click.option("--do_table/--no_table", default=False, help="Do table of results.")
@click.option("--do_figure/--no_figure", default=False, help="Do figure of results.")
def main(do_table, do_figure):
    if do_table:
        print("Values for CIFAR10")
        get_table(db_name="cifar10")
        print("Values for CIFAR100")
        get_table(db_name="cifar100")

    if do_figure:
        print("Working on CIFAR10")
        plot_results(db_name="cifar10", debug=False)
        print("Working on CIFAR100")
        plot_results(db_name="cifar100", debug=False)


if __name__ == "__main__":
    main()
