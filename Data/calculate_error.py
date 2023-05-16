import copy
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

"""
For errors: Only call calculate_error
type_error: "all_map", "peaks", "action_zone"
X_test: available points on the lake map, points where ASVs may travel (black part of the map), (x, y)
benchmark_array: vector containing the ground truth value of the X_test coordinates, benchmark[i]=f(X_test(i))
vehicles: number of vehicles in the fleet
mu: mean estimated by your gp
error_peak: vector of errors of the peaks
action_mse: vector of mse of action zones
map_mse: vector of the mse of the whole map

For plots: plot_classic
mu: mean estimated by your gp
sigma: uncertainty estimated by your gp
part_ant: matrix with the positions through which the vehicles traveled [-1, 2 * number of vehicles] [x1, y1, x2, y2, ...]
grid: map of the lake in 0 and 1
X_test: available points on the lake map, points where ASVs may travel (black part of the map), (x, y)

print_error -> to have the average of errors and mse of all simulations and to save the data in excel documents
error_peak: vector of errors of the peaks
action_mse: vector of mse of action zones
map_mse: vector of the mse of the whole map
"""


class DetectContaminationAreas():
    def __init__(self, X_test, benchmark, vehicles=4, area=100):
        self.coord = copy.copy(X_test)
        self.coord_bench = copy.copy(X_test)
        self.coord_real = copy.copy(X_test)
        self.radio = area / vehicles
        self.benchmark = copy.copy(benchmark)
        self.ava = np.array(X_test)
        self.vehicles = vehicles

    def benchmark_areas(self):
        dict_coord_bench = {}
        dict_index_bench = {}
        dict_impor_bench = {}
        dict_bench_ = {}
        dict_limits_bench = {}
        array_action_zones_bench = list()
        coordinate_action_zones_bench = list()
        bench = copy.copy(self.benchmark)
        index_xtest = list()
        index_center_bench = list()
        j = 0
        array_max_x_bench = list()
        array_max_y_bench = list()
        max_bench_list = list()
        action_zone_bench = list()
        warning_bench = max(bench) * 0.33
        impo = self.vehicles * 10 + 10
        cen = 0

        for i in range(len(bench)):
            if bench[i] >= warning_bench:
                array_action_zones_bench.append(bench[i])
                coordinate_action_zones_bench.append(self.coord_bench[i])
                index_xtest.append(i)
        while cen < self.vehicles:
            max_action_zone_bench = max(array_action_zones_bench)
            max_index_bench = array_action_zones_bench.index(max_action_zone_bench)
            index_center_bench.append(index_xtest[max_index_bench])
            max_coordinate_bench = coordinate_action_zones_bench[max_index_bench]
            max_bench_list.append(max_action_zone_bench)
            x_max_bench = max_coordinate_bench[0]
            array_max_x_bench.append(x_max_bench)
            y_max_bench = max_coordinate_bench[1]
            array_max_y_bench.append(y_max_bench)
            coordinate_array = np.array(coordinate_action_zones_bench)
            m = 0
            for i in range(len(array_action_zones_bench)):
                if math.sqrt(
                        (x_max_bench - coordinate_array[i, 0]) ** 2 + (
                                y_max_bench - coordinate_array[i, 1]) ** 2) <= self.radio:
                    index_del = i - m
                    del array_action_zones_bench[index_del]
                    del coordinate_action_zones_bench[index_del]
                    del index_xtest[index_del]
                    m += 1
            if len(array_action_zones_bench) == 0:
                break
            cen += 1
        center_peaks_bench = np.column_stack((array_max_x_bench, array_max_y_bench))
        for w in range(len(array_max_x_bench)):
            list_zone_bench = list()
            list_coord_bench = list()
            list_impo = list()
            del_list = list()
            coordinate_array = np.array(self.coord_bench)
            for i in range(len(self.coord_bench)):
                if math.sqrt(
                        (array_max_x_bench[w] - coordinate_array[i, 0]) ** 2 + (
                                array_max_y_bench[w] - coordinate_array[i, 1]) ** 2) <= self.radio:
                    list_zone_bench.append(bench[i])
                    list_coord_bench.append(self.coord_bench[i])
                    list_impo.append(impo)
                    del_list.append(i)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del self.coord_bench[index_del]
                m += 1
            array_list_coord = np.array(list_coord_bench)
            x_coord = array_list_coord[:, 0]
            y_coord = array_list_coord[:, 1]
            max_x_coord = max(x_coord)
            min_x_coord = min(x_coord)
            max_y_coord = max(y_coord)
            min_y_coord = min(y_coord)
            dict_limits_bench["action_zone%s" % j] = [min_x_coord, max_y_coord, max_x_coord, min_y_coord]
            index = list()
            for i in range(len(array_list_coord)):
                x = array_list_coord[i, 0]
                y = array_list_coord[i, 1]
                for p in range(len(self.ava)):
                    if x == self.ava[p, 0] and y == self.ava[p, 1]:
                        index.append(p)
                        action_zone_bench.append(self.benchmark[p])
                        break
            dict_bench_["action_zone%s" % j] = list_zone_bench
            dict_coord_bench["action_zone%s" % j] = list_coord_bench
            dict_index_bench["action_zone%s" % j] = index
            dict_impor_bench["action_zone%s" % j] = list_impo
            impo -= 10
            j += 1
            """Key variables:
            dict_bench_: dictionary containing the gt values of the coordinates of the action zone [i].
            action_zone_bench: gt values in the coordinates of the action zones. The values of the action zones are 
            together (values of zone 1, values of zone 2, ...) (append function).
            dict_index_bench: dictionary containing the indexes of the coordinates found in the action zone [i]
            max_bench_list: list of the peak values of the action zones, position [i] peak of action zone [i]
            index_center_bench: indexes of contamination peaks in the action zones, position [i] peak of action zone [i]
            """
        return j, dict_index_bench, dict_bench_, dict_coord_bench, center_peaks_bench, max_bench_list, \
               dict_limits_bench, action_zone_bench, dict_impor_bench, index_center_bench


def calculate_error(type_error, X_test, bench_array, vehicles, mu, error_peak, action_mse, map_mse):
    detect = DetectContaminationAreas(X_test, bench_array, vehicles=vehicles, area=100)
    j, dict_index_bench, dict_bench_, dict_coord_bench, center_peaks_bench, max_bench_list, \
    dict_limits_bench, action_zone_bench, dict_impor_bench, index_center_bench = detect.benchmark_areas()

    dict_error_peak = {}
    dict_bench = {}
    dict_error = {}
    if type_error == 'all_map':
        error = mean_squared_error(y_true=bench_array, y_pred=mu)
        map_mse.append(error)
    elif type_error == 'peaks':
        peak_mean = list()
        for i in range(len(index_center_bench)):
            max_az = mu[index_center_bench[i]]
            dict_error_peak["action_zone%s" % i] = abs(max_bench_list[i] - max_az)
            peak_mean.append(abs(max_bench_list[i] - max_az))
        error_peak.append(np.mean(np.array(peak_mean)))
    elif type_error == 'action_zone':
        estimated_all = list()
        mse_action = list()
        for i in range(len(center_peaks_bench)):
            bench_action = copy.copy(dict_bench["action_zone%s" % i])
            estimated_action = list()
            index_action = copy.copy(dict_index_bench["action_zone%s" % i])
            for j in range(len(index_action)):
                value = mu[index_action[j]]
                estimated_action.append(value[0])
                estimated_all.append(mu[index_action[j]])
            error_action = mean_squared_error(y_true=bench_action, y_pred=estimated_action)
            dict_error["action_zone%s" % i] = copy.copy(error_action)
            mse_action.append(error_action)
        action_mse.append(np.mean(np.array(mse_action)))
    return error_peak, action_mse, map_mse


def print_error(error_peak, action_mse, map_mse):
    print("Error peak:", np.mean(np.array(error_peak)), '+-', np.std(np.array(error_peak)) * 1.96)
    print("MSE action:", np.mean(np.array(action_mse)), '+-', np.std(np.array(action_mse)) * 1.96)
    print("MSE map:", np.mean(np.array(map_mse)), '+-', np.std(np.array(map_mse)) * 1.96)
    df1 = pd.DataFrame(error_peak)
    df1.to_excel('../Test/Results/Error/ErrorLawnmower.xlsx')
    df2 = pd.DataFrame(action_mse)
    df2.to_excel('../Test/Results/MSEAZ/MSEAZLawnmower.xlsx')
    df3 = pd.DataFrame(map_mse)
    df3.to_excel('../Test/Results/MSEM/MSEMLawnmower.xlsx')


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
