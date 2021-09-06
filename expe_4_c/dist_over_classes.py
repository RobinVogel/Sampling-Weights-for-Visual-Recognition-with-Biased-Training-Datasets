import click
import numpy as np
import matplotlib.pyplot as plt


def plot_natural_class_assignment():
    plt.figure(figsize=(4, 4))
    with open("emp_obs_per_bin/last_class_assignment.txt", "rt") as f:
        for i, line in enumerate(f.read().strip().split("\n")):
            nos = [int(a) for a in line.split(" ")]
            plt.plot(range(0, len(nos)), nos, label="D{}".format(i))

    plt.legend()
    plt.grid()
    plt.savefig("figures/bar_assign_class.pdf")


def plot_natural_assignment():
    with open("emp_obs_per_bin/last_assignment.txt", "rt") as f:
        nos = [int(a) for a in f.read().strip().split("\n")]

    plt.figure(figsize=(4, 4))
    plt.bar(range(0, len(nos)), nos)
    plt.grid()
    plt.savefig("figures/bar_assign.pdf")


def get_distribution(alpha, outname, K=8, n_tot=50000):
    weights = [(alpha ** i) for i in range(0, K)]
    weights = np.array(weights) / np.sum(weights)
    cardinals = [str(int(n_tot * weights[i])) for i in range(0, K)]

    with open(outname, "wt") as f:
        f.write("\n".join(cardinals))


@click.command()
@click.option(
    "--do_plot/--no_plot", default=False, help="Do some plot of distributions."
)
@click.option("--alpha", default=0.5, help="Power law factor.")
@click.option("--outfile", default="obs_per_bin/power_law.txt", help="Output name.")
def main(do_plot, alpha, outfile):
    if do_plot:
        plot_natural_class_assignment()
        plot_natural_assignment()
    get_distribution(alpha, outfile)


if __name__ == "__main__":
    main()
