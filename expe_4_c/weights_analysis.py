import numpy as np
import matplotlib.pyplot as plt


def main():
    plot_1 = "cifar10_gamma_0.1_ite0_concat"  # "cifar10_gamma_1_alpha_0.75_ite0"
    plot_2 = plot_1  # "cifar100_gamma_10_ite0_concat"
    plot_3 = plot_1
    # "cifar10_gamma_1_alpha_0.75_ite0_concat"
    # "cifar10_gamma_1_ite0_concat"  # "cifar10_gamma_1_ite0"
    for cplot in [plot_1, plot_2, plot_3]:
        weights = np.zeros(50000)
        with open("weights/{}.txt".format(cplot), "rt") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                index = int(line.split(" ")[0])
                val = float(line.split(" ")[1])
                weights[index] = val
        weights /= weights.sum()
        print("{} {} {}".format(cplot, weights.max(), weights.min()))
        print("num < 10-8 = {}".format((weights < 1e-8).sum()))
        plt.figure(figsize=(10, 4))
        plt.hist(weights, bins=250)
        plt.savefig("figures/{}_weights.pdf".format(cplot))


if __name__ == "__main__":
    main()
