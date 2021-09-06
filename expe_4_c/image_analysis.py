import json
import itertools

import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


from torchvision.datasets import CIFAR10, CIFAR100


@click.command()
@click.option("--name_db", default="cifar10", help="Name of the original db.")
@click.option("--colors/--no_colors", default=False, help="Do color analysis.")
@click.option("--black/--no_black", default=False, help="Do black analysis.")
@click.option("--borders/--no_borders", default=False, help="Do border analysis.")
def main(name_db, colors, black, borders):
    if name_db == "cifar10":
        torch_data = CIFAR10("./data", download=True)
    elif name_db == "cifar100":
        torch_data = CIFAR100("./data", download=True)
    X = torch_data.data
    y = torch_data.targets

    if colors:
        avg_colors = X.mean(axis=(1, 2))
        var_colors = X.var(axis=(1, 2))

        for type_plot in ["avg", "var"]:
            for dim1, dim2 in itertools.combinations(range(0, 3), 2):
                plt.figure()
                if type_plot == "avg":
                    data = avg_colors
                elif type_plot == "var":
                    data = var_colors
                else:
                    raise ValueError("Unknown stuff.")
                plt.scatter(data[:, dim1], data[:, dim2], alpha=0.2)
                # sns.kdeplot(x=data[:, dim1], y=data[:, dim2])
                plt.grid()
                plt.xlabel("dim {}".format(dim1))
                plt.ylabel("dim {}".format(dim2))
                savepath = "figures/exploratory/{}_{}_vs_{}.pdf".format(
                    type_plot, dim1, dim2
                )
                plt.savefig(savepath)
                plt.close()

    if black:
        plt.figure()
        plt.hist(avg_colors.mean(axis=1))
        # sns.kdeplot(x=data[:, dim1], y=data[:, dim2])
        plt.grid()
        plt.xlabel("intensity")
        plt.ylabel("prop")
        savepath = "figures/exploratory/bw.pdf"
        plt.savefig(savepath)
        plt.close()

    if borders:
        hsv = True
        if hsv:
            new_X = list()
            for x in X:
                new_X.append(matplotlib.colors.rgb_to_hsv(x))
            new_X = np.array(new_X)
            new_X[:, :, :, 2] = new_X[:, :, :, 2] / 255
            X = new_X
        sborder = 2
        border1, border2 = X[:, :sborder, :, :], X[:, :, :sborder, :]
        border3, border4 = (
            X[:, (32 - sborder) :, :, :],
            X[:, :, (32 - sborder) :, :],
        )

        # print(border1.shape)
        # print(border2.shape)
        # print(border3.shape)
        # print(border2.shape)

        all_borders = np.concatenate(
            [border1, border2.swapaxes(1, 2), border3, border4.swapaxes(1, 2)], axis=1
        )
        avg_border_color = all_borders.mean(axis=(1, 2))
        import ipdb

        ipdb.set_trace()

        # avg_border_color = (
        #     border1.var(axis=(1, 2))
        #     + border2.mean(axis=(1, 2))
        #     + border3.mean(axis=(1, 2))
        #     + border4.mean(axis=(1, 2))
        # ) / 4

        print(avg_border_color.shape)
        n_elems = 1000

        for dim1, dim2 in itertools.combinations(range(0, 3), 2):
            plt.figure()
            inds = np.random.choice(range(0, avg_border_color.shape[0]), size=n_elems)
            plt.scatter(
                avg_border_color[inds, dim1], avg_border_color[inds, dim2], alpha=0.2
            )
            # sns.kdeplot(x=data[:, dim1], y=data[:, dim2])
            plt.grid()
            plt.xlabel("dim {}".format(dim1))
            plt.ylabel("dim {}".format(dim2))
            savepath = "figures/exploratory/{}_{}_vs_{}.pdf".format(
                "border", dim1, dim2
            )
            plt.savefig(savepath)
            plt.close()

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(
            avg_border_color[inds, 0],
            avg_border_color[inds, 1],
            avg_border_color[inds, 2],
        )
        plt.xlabel("H")
        plt.ylabel("S")

        print("max x : ", np.max(avg_border_color[inds, 0]))
        print("max y : ", np.max(avg_border_color[inds, 1]))
        print("max z : ", np.max(avg_border_color[inds, 2]))
        plt.show()

        plt.figure()
        inds = np.random.choice(range(0, avg_border_color.shape[0]), size=n_elems)
        plt.hist(all_borders.std(axis=(1, 2))[:, 2], bins=100)

        # sns.kdeplot(x=data[:, dim1], y=data[:, dim2])
        plt.grid()
        savepath = "figures/exploratory/hist_std_V.pdf"
        plt.savefig(savepath)
        plt.close()


if __name__ == "__main__":
    main()
