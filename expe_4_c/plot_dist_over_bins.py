import numpy as np
import matplotlib.pyplot as plt


def main():

    fig, subplots = plt.subplots(1, 3, figsize=(6, 2))
    x = range(1, 9)
    with open("emp_obs_per_bin/last_assignment.txt", "rt") as f:
        y_emp = [int(a) for a in f.read().strip().split("\n")]
    with open("obs_per_bin/balanced.txt", "rt") as f:
        y_bal = [int(a) for a in f.read().strip().split("\n")]
    with open("obs_per_bin/alpha_0.75.txt", "rt") as f:
        y_lgt = [int(a) for a in f.read().strip().split("\n")]

    subplots[0].bar(x, y_emp, label="empirical")  #  , color="darkgrey")
    subplots[1].bar(x, y_bal, label="balanced")  # , color="darkgrey")
    subplots[2].bar(x, y_lgt, label="long tail")  #  , color="darkgrey")
    titles = ["empirical", "balanced", "long-tail"]
    for coord in range(0, 3):
        subplots[coord].grid()
        # subplots[coord].xlabel("# obs")
        # subplots[coord].ylabel("Dataset")
        subplots[coord].set_xticks(x, [""] * 8)
        subplots[coord].set_title(titles[coord])
    # subplots[coord].yticks()

    plt.xticks(x, x)

    # plt.ylabel("")
    # plt.xlabel("")
    plt.tight_layout()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in subplots.flat:
        ax.label_outer()

    fig.text(0.5, 0.005, "Dataset", ha="center")
    fig.text(0.001, 0.5, "# obs", va="center", rotation="vertical")

    plt.savefig("figures/dist_over_bins.pdf")


if __name__ == "__main__":
    main()
