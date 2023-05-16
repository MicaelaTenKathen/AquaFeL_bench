import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


def plot_classic(mu, sigma, part_ant, grid, X_test):
    cmap1 = LinearSegmentedColormap.from_list('name', ['peachpuff', 'tomato', 'maroon'])
    cmap = LinearSegmentedColormap.from_list('name', ['green', 'yellow', 'red'])
    cmap2 = LinearSegmentedColormap.from_list('name', ['red', 'purple'])
    cmap3 = LinearSegmentedColormap.from_list('name', ['olive', 'cadetblue'])
    cmap4 = LinearSegmentedColormap.from_list('name', ['grey', 'navy'])
    cmap5 = LinearSegmentedColormap.from_list('name', ['darkviolet', 'crimson'])
    cmap6 = LinearSegmentedColormap.from_list('name', ['lime', 'gold'])
    colors = ['winter', 'copper', cmap2, 'spring', 'cool', cmap3, 'autumn', cmap4, cmap5,
                   cmap6]

    def Z_var_mean(mu, sigma):
        Z_un = np.zeros([grid.shape[0], grid.shape[1]])
        Z_mean = np.zeros([grid.shape[0], grid.shape[1]])
        for i in range(len(X_test)):
            Z_un[X_test[i][0], X_test[i][1]] = sigma[i]
            Z_mean[X_test[i][0], X_test[i][1]] = mu[i]
        Z_un[grid == 0] = np.nan
        Z_mean[grid == 0] = np.nan
        return Z_un, Z_mean

    def plot_trajectory_classic(ax, x, y, z=None, colormap='jet', linewidth=1, plot_waypoints=True,
                                markersize=0.5):
        if z is None:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
            lc = LineCollection(segments, norm=plt.Normalize(0, 1), cmap=plt.get_cmap(colormap), linewidth=linewidth)
            lc.set_array(np.linspace(0, 1, len(x)))
            ax.add_collection(lc)
            if plot_waypoints:
                ax.plot(x, y, '.', color='black', markersize=markersize)
        else:
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
            lc = Line3DCollection(segments, norm=plt.Normalize(0, 1), cmap=plt.get_cmap(colormap), linewidth=linewidth)
            lc.set_array(np.linspace(0, 1, len(x)))
            ax.add_collection(lc)
            ax.scatter(x, y, z, 'k')
            if plot_waypoints:
                ax.plot(x, y, 'kx')

    Z_var, Z_mean = Z_var_mean(mu, sigma)
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    initial_x = list()
    initial_y = list()
    final_x = list()
    final_y = list()
    for i in range(part_ant.shape[1]):
        if i % 2 == 0:
            initial_x.append(part_ant[0, i])
            final_x.append(part_ant[-1, i])
        else:
            initial_y.append(part_ant[0, i])
            final_y.append(part_ant[-1, i])

    vehicles = int(part_ant.shape[1] / 2)
    for i in range(vehicles):
        plot_trajectory_classic(axs[0], part_ant[:, 2 * i], part_ant[:, 2 * i + 1], colormap=colors[i])

    axs[0].plot(initial_x, initial_y, 'o', color='black', markersize=3, label='ASVs initial positions')
    axs[0].plot(final_x, final_y, 'X', color='red', markersize=3, label='ASVs final positions')
    axs[0].legend(loc=3, fontsize=6)

    im2 = axs[0].imshow(Z_var.T, interpolation='bilinear', origin='lower', cmap="gist_yarg", vmin=0, vmax=1.0)
    plt.colorbar(im2, ax=axs[0], label='σ', shrink=1.0)
    # axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
    axs[0].set_xticks([0, 50, 100])
    axs[0].set_aspect('equal')
    axs[0].set_ylim([150, 0])
    axs[0].grid(True)
    ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
    axs[0].xaxis.set_major_formatter(ticks_x)

    ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
    axs[0].yaxis.set_major_formatter(ticks_y)

    im3 = axs[1].imshow(Z_mean.T, interpolation='bilinear', origin='lower', cmap='jet', vmin=0, vmax=1.0)
    plt.colorbar(im3, ax=axs[1], label='µ', shrink=1.0)
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
    axs[1].set_xticks([0, 50, 100])
    axs[1].set_ylim([150, 0])
    axs[1].set_aspect('equal')
    axs[1].grid(True)
    ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
    axs[1].xaxis.set_major_formatter(ticks_x)

    ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
    axs[1].yaxis.set_major_formatter(ticks_y)

    # plt.savefig("../Image/Contamination/GT3/Tabla_3.png")
    plt.show()