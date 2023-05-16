import openpyxl

from Data.limitsn import Limits
from Environment.map import Map
from Benchmark.benchmark_functions import Benchmark_function
from Environment.boundsn import Bounds
from Data.utils import Utils
from Environment.contamination_areas import DetectContaminationAreas
from Environment.plot import Plots

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import random
import math
import gym
import copy


class RandomPPEnvironment():

    def __init__(self, ys, resolution, vehicles, initial_seed, initial_position, exploration_distance, type_error):
        self.resolution = resolution
        self.exploration_distance = exploration_distance
        self.file = 'Test/Elsevier/Lawnmower'
        self.type_error = type_error
        self.error_peak = list()
        self.action_mse = list()
        self.map_mse = list()
        self.initial_type_error = type_error
        self.xs = int(10000 / (15000 / ys))
        self.ys = ys
        self.n_data = 0
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1e-6)  # optimizer=None)
        self.grid_or = Map(self.xs, ys).black_white()

        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys
        self.tim = 0
        if self.tim == 0:
            self.df_bounds, self.X_test_no, self.bench_limits_no = Bounds(self.resolution, self.xs, self.ys,
                                                                          load_file=False).map_bound()
            self.X_test, self.bench_limits = Bounds(self.resolution, self.xs, self.ys,
                                                    load_file=False).available_xtest()
            # = Bounds(self.resolution, self.xs, self.ys,
            #                                                    load_file=False).available_xtest()
            self.tim = 1
        self.secure = Bounds(self.resolution, self.xs, self.ys).interest_area()
        self.df_bounds_x = Bounds(self.resolution, self.xs, self.ys).bounds_y()
        self.vehicles = vehicles
        self.util = Utils(self.vehicles)
        self.seed = initial_seed
        self.initial_seed = initial_seed
        self.initial_position = initial_position
        self.vel = 10
        self.limits = Limits(self.secure, self.xs, self.ys, self.vehicles)
        self.dict_direction = {}
        self.dict_turn = {}
        self.asv = 1
        self.check = True
        self.turn = False
        self.g = 0
        self.post_array = np.ones(self.vehicles)
        self.distances = np.zeros(self.vehicles)
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.x_h = []
        self.y_h = []
        self.water_samples = []
        self.dict_n_pos = {}
        self.dict_c_pos = {}
        self.dict_error_comparison = {}
        self.x_bench = None
        self.y_bench = None
        self.mu = []
        self.sigma = []
        self.samples = 0
        self.dict_error = {}
        self.dict_error_peak = {}
        self.dict_error_az = {}
        self.type_error = self.initial_type_error
        self.error = []
        self.ERROR_data = []
        self.error_data = []
        self.dict_check_turn = {}
        self.it = []
        self.f = 0
        self.lam = 0.375
        self.last_sample = 0
        self.initial = True
        self.save = 0
        self.save_dist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                          500, 525, 550, 575, 600, 625, 650, 675, 700]

    def reset(self):
        self.seed += 1
        self.bench_function, self.bench_array, self.num_of_peaks, self.index_a = Benchmark_function(self.grid_or,
                                                                                                    self.resolution,
                                                                                                    self.xs, self.ys,
                                                                                                    self.X_test,
                                                                                                    self.seed,
                                                                                                    self.vehicles).create_new_map()
        random.seed(self.seed)
        self.detect_areas = DetectContaminationAreas(self.X_test, self.bench_array, vehicles=self.vehicles,
                                                     area=self.xs)
        self.centers_bench, self.dict_index_bench, self.dict_bench, self.dict_coord_bench, self.center_peaks_bench, \
        self.max_bench_list, self.dict_limits_bench, self.action_zone_bench, self.dict_impo_bench, \
        self.index_center_bench = self.detect_areas.benchmark_areas()
        self.reset_variables()
        self.first_values()

    def reset_variables(self):
        self.dict_direction = {}
        self.dict_turn = {}
        self.dict_check_turn = {}
        self.asv = 1
        self.check = True
        self.turn = False
        self.g = 0
        self.post_array = np.ones(self.vehicles)
        self.distances = np.zeros(self.vehicles)
        self.part_ant = np.zeros((1, self.vehicles * 2))
        self.array_part = np.zeros((1, self.vehicles * 2))
        self.x_h = []
        self.y_h = []
        self.water_samples = []
        self.dict_n_pos = {}
        self.dict_c_pos = {}
        self.dict_error_comparison = {}
        self.x_bench = None
        self.y_bench = None
        self.mu = []
        self.sigma = []
        self.samples = 0
        self.dict_error = {}
        self.dict_error_peak = {}
        self.dict_error_az = {}
        self.type_error = self.initial_type_error
        self.error = []
        self.ERROR_data = []
        self.error_data = []
        self.it = []
        self.f = 0
        self.lam = 0.375
        self.last_sample = 0
        self.initial = True
        self.save = 0
        self.save_dist = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
                          500, 525, 550, 575, 600, 625, 650, 675, 700]

    # def moving_direction(self):
    #     for i in range(self.vehicles):
    #         if i % 2 == 0:
    #             if (self.bench_limits[1] - self.initial_position[i, 0]) < (
    #                     self.initial_position[i, 0] - self.bench_limits[0]):
    #                 self.dict_direction["vehicle%s" % i] = [-1, 0]
    #             else:
    #                 self.dict_direction["vehicle%s" % i] = [1, 0]
    #         else:
    #             if (self.bench_limits[3] - self.initial_position[i, 1]) < (
    #                     self.initial_position[i, 1] - self.bench_limits[2]):
    #                 self.dict_direction["vehicle%s" % i] = [0, -1]
    #             else:
    #                 self.dict_direction["vehicle%s" % i] = [0, 1]
    #
    # def moving_turn(self, i, dfirst):
    #     if dfirst:
    #         self.dict_check_turn["vehicle%s" % i] = False
    #         if i % 2 == 0:
    #             x_turn = self.dict_direction["vehicle%s" % i][0]
    #             if (self.bench_limits[3] - self.initial_position[i, 1]) < (
    #                     self.initial_position[i, 1] - self.bench_limits[2]):
    #                 self.dict_turn["vehicle%s" % i] = [x_turn, -1]
    #             else:
    #                 self.dict_turn["vehicle%s" % i] = [x_turn, 1]
    #         else:
    #             y_turn = self.dict_direction["vehicle%s" % i][1]
    #             if (self.bench_limits[1] - self.initial_position[i, 0]) < (
    #                     self.initial_position[i, 0] - self.bench_limits[0]):
    #                 self.dict_turn["vehicle%s" % i] = [-1, y_turn]
    #             else:
    #                 self.dict_turn["vehicle%s" % i] = [1, y_turn]
    #     else:
    #         self.dict_check_turn["vehicle%s" % i] = True
    #
    # def move_vehicle(self, c_pos, vehicle):
    #     direction = self.dict_direction["vehicle%s" % vehicle]
    #     n_pos = list(map(lambda x, y, z: x + y * z, c_pos, direction, self.vel))
    #     self.check = self.limits.check_lm_limits(n_pos, vehicle)
    #     if not self.check:
    #         n_pos = list(map(lambda x, y, z: x * y + z, self.dict_turn["vehicle%s" % vehicle], self.vel,
    #                          c_pos))
    #         self.dict_direction["vehicle%s" % vehicle] = list(
    #             map(lambda x, y: x * y, self.dict_direction["vehicle%s" % vehicle], [-1, -1]))
    #         self.moving_turn(vehicle, dfirst=False)
    #
    #     return n_pos

    # def moving_direction(self):
    #     for i in range(self.vehicles):
    #         # if i % 2 == 0:
    #         if (self.bench_limits_no[1] - self.initial_position[i, 0]) <= (
    #                 self.initial_position[i, 0] - self.bench_limits_no[0]):
    #             self.dict_direction["vehicle%s" % i] = [-1, 0]
    #         else:
    #             self.dict_direction["vehicle%s" % i] = [1, 0]
            # else:
            #     if (self.bench_limits[3] - self.initial_position[i, 1]) <= (
            #             self.initial_position[i, 1] - self.bench_limits[2]):
            #         self.dict_direction["vehicle%s" % i] = [0, -1]
            #     else:
            #         self.dict_direction["vehicle%s" % i] = [0, 1]

    # def moving_turn(self, i, dfirst=False):
    #     if dfirst:
    #         self.dict_check_turn["vehicle%s" % i] = False
    #         # if i % 2 == 0:
    #         x_turn = self.dict_direction["vehicle%s" % i][0]
    #         if (self.bench_limits_no[3] - self.initial_position[i, 1]) <= (
    #                 self.initial_position[i, 1] - self.bench_limits_no[2]):
    #             self.dict_turn["vehicle%s" % i] = [0, -1]
    #         else:
    #             self.dict_turn["vehicle%s" % i] = [0, 1]
    #         # else:
    #         #     y_turn = self.dict_direction["vehicle%s" % i][1]
    #         #     if (self.bench_limits[1] - self.initial_position[i, 0]) <= (
    #         #             self.initial_position[i, 0] - self.bench_limits[0]):
    #         #         self.dict_turn["vehicle%s" % i] = [-1, 0]
    #         #     else:
    #         #         self.dict_turn["vehicle%s" % i] = [1, 0]
    #     else:
    #         self.dict_check_turn["vehicle%s" % i] = True

    # def move_vehicle(self, c_pos, vehicle):
    #     direction = copy.copy(self.dict_direction["vehicle%s" % vehicle])
    #     n_pos = copy.copy(list(map(lambda x, y, z: x + y * z, c_pos, direction, self.vel)))
    #     self.check = self.limits.check_lm_limits(n_pos, vehicle)
    #     if not self.check:
    #         n_pos = copy.copy(list(map(lambda x, y, z: x * y + z, self.dict_turn["vehicle%s" % vehicle], self.vel,
    #                          c_pos)))
    #         self.dict_direction["vehicle%s" % vehicle] = copy.copy(list(
    #             map(lambda x, y: x * y, self.dict_direction["vehicle%s" % vehicle], [-1, -1])))
    #         self.moving_turn(vehicle, dfirst=True)
    #         self.check2 = self.limits.check_lm_limits(n_pos, vehicle)
    #         if not self.check2:
    #             n_pos = copy.copy(list(map(lambda w, x, y, z: (w + x) * y + z, self.dict_turn["vehicle%s" % vehicle], self.dict_direction["vehicle%s" % vehicle], self.vel,
    #                                        c_pos)))
            # for i, subfleet in enumerate(self.sub_fleets):
            #     sensors = self.s_sf[i]
            #     for s, sensor in enumerate(sensors):
            #         bench = copy.copy(self.dict_benchs_[sensor]['map_created'])
            #         self.plot.benchmark(bench, sensor)
            #         mu = copy.copy(self.dict_sensors_[sensor]['mu']['data'])
            #         sigma = copy.copy(self.dict_sensors_[sensor]['sigma']['data'])
            #         vehicles = copy.copy(self.dict_sensors_[sensor]['vehicles'])
            #         trajectory = list()
            #         first = True
            #         list_ind = list()
            #         for veh in vehicles:
            #             list_ind.append(self.P.nodes[veh]['index'])
            #             if first:
            #                 trajectory = np.array(self.P.nodes[veh]['U_p'])
            #                 first = False
            #             else:
            #                 new = np.array(self.P.nodes[veh]['U_p'])
            #                 trajectory = np.concatenate((trajectory, new), axis=1)
            #         self.plot.plot_classic(mu, sigma, trajectory, sensor, list_ind)
        # return n_pos

    def move_vehicle(self, c_pos, vehicle):
        angle, distance = self.dict_direction["vehicle%s" % vehicle]
        self.check = False
        while not self.check:
            x = c_pos[0] + round(distance) * math.cos(angle)
            y = c_pos[1] + round(distance) * math.sin(angle)
            n_pos = [int(x), int(y)]
            self.check = self.limits.check_lm_limits(n_pos, vehicle)
            if not self.check:
                # angle = 2 * math.pi * random.random()
                value = random.randint(0, 4)
                angle = 2 * math.pi * (value / 4)
                self.dict_direction["vehicle%s" % vehicle][0] = angle
        return n_pos

    def next_measure_point(self):
        for i in range(self.vehicles):
            # angle = 2 * math.pi * random.random()
            value = random.randint(0, 4)
            angle = 2 * math.pi * (value / 4)
            distance = np.min(self.post_array) * self.lam
            self.dict_direction["vehicle%s" % i] = [angle, distance]

    def take_sample(self, n_pos):
        self.x_bench = n_pos[0]
        self.y_bench = n_pos[1]
        sample_value = [self.bench_function[self.x_bench][self.y_bench]]
        return sample_value

    def check_duplicate(self, n_pos, sample_value):
        self.duplicate = False
        for i in range(len(self.x_h)):
            if self.x_h[i] == self.x_bench and self.y_h[i] == self.y_bench:
                self.duplicate = True
                self.water_samples[i] = sample_value
                break
            else:
                self.duplicate = False
        if self.duplicate:
            pass
        else:
            self.x_h.append(int(n_pos[0]))
            self.y_h.append(int(n_pos[1]))
            self.water_samples.append(sample_value)

    def gp_regression(self):

        """
        Fits the gaussian process.
        """

        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(self.water_samples).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        self.mu, self.sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = np.min(np.exp(self.gpr.kernel_.theta[0]))
        r = self.n_data
        self.post_array[r] = post_ls

        return self.post_array

    def calculate_error(self):
        if self.type_error == 'all_map':
            self.error = mean_squared_error(y_true=self.bench_array, y_pred=self.mu)
        elif self.type_error == 'peaks':
            peak_error = []
            for i in range(len(self.index_center_bench)):
                max_az = self.mu[self.index_center_bench[i]]
                self.dict_error_peak["action_zone%s" % i] = abs(self.max_bench_list[i] - max_az)
                peak_error.append(abs(self.max_bench_list[i] - max_az))
                # self.error_peak.append(abs(self.max_bench_list[i] - max_az))
            self.error_peak.append(np.mean(np.array(peak_error)))
        elif self.type_error == 'action_zone':
            action_mse = []
            estimated_all = list()
            for i in range(len(self.center_peaks_bench)):
                bench_action = copy.copy(self.dict_bench["action_zone%s" % i])
                estimated_action = list()
                index_action = copy.copy(self.dict_index_bench["action_zone%s" % i])
                for j in range(len(index_action)):
                    value = self.mu[index_action[j]]
                    estimated_action.append(value[0])
                    estimated_all.append(self.mu[index_action[j]])
                error_action = mean_squared_error(y_true=bench_action, y_pred=estimated_action)
                self.dict_error["action_zone%s" % i] = copy.copy(error_action)
                action_mse.append(error_action)
                # self.action_mse.append(error_action)
            # self.error = mean_squared_error(y_true=self.action_zone_bench, y_pred=estimated_all)
            self.action_mse.append(np.mean(np.array(action_mse)))
        return self.error

    def save_data(self):
        if self.save < (self.exploration_distance / 25):
            mult = self.save_dist[self.save]
            mult_min = mult - 5
            mult_max = mult + 5
            if mult_min <= np.max(self.distances) < mult_max:
                if self.seed == self.initial_seed + 1:
                    if self.initial:
                        for i in range(len(self.save_dist)):
                            self.dict_error_comparison["Distance%s" % i] = list()
                        self.initial = False
                error_list = copy.copy(self.dict_error_comparison["Distance%s" % self.save])
                self.ERROR_data = self.calculate_error()
                error_list.append(self.ERROR_data)
                self.dict_error_comparison["Distance%s" % self.save] = copy.copy(error_list)
                self.save += 1

    def save_excel(self):
        for i in range(int(self.exploration_distance / 25)):
            wb = openpyxl.Workbook()
            hoja = wb.active
            hoja.append(self.dict_error_comparison["Distance%s" % i])
            wb.save('../Test/' + self.file + '/ALLCONError_' + str(self.save_dist[i]) + '.xlsx')

    def first_values(self):
        for i in range(self.vehicles):
            self.dict_c_pos["vehicle%s" % i] = list(self.initial_position[i])
            sample_value = self.take_sample(self.dict_c_pos["vehicle%s" % i])
            n_pos = self.dict_c_pos["vehicle%s" % i]
            self.dict_n_pos["vehicle%s" % i] = n_pos
            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, n_pos, self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.check_duplicate(n_pos, sample_value)
            self.dict_c_pos["vehicle%s" % i] = n_pos

            self.samples += 1

            self.n_data += 1
            if self.n_data > self.vehicles - 1:
                self.n_data = 0

            self.post_array = self.gp_regression()
        self.next_measure_point()
        self.it.append(self.g)

        self.k = self.vehicles
        self.ok = False

    def simulator(self):
        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.max(self.distances)
        self.n_data = 0
        self.f += 1
        while dis_steps < 10:

            previous_dist = np.max(self.distances)

            for i in range(self.vehicles):
                n_pos = self.move_vehicle(self.dict_c_pos["vehicle%s" % i], i)
                self.dict_n_pos["vehicle%s" % i] = n_pos
                self.dict_c_pos["vehicle%s" % i] = n_pos

            for i in range(self.vehicles):
                n_pos = self.dict_n_pos["vehicle%s" % i]
                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, n_pos, self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.vehicles - 1:
                    self.n_data = 0

            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam):
                self.k += 1
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for i in range(self.vehicles):
                    n_pos = self.dict_n_pos["vehicle%s" % i]
                    sample_value = self.take_sample(n_pos)
                    self.check_duplicate(n_pos, sample_value)

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > self.vehicles - 1:
                        self.n_data = 0

                    self.post_array = self.gp_regression()
                self.next_measure_point()
                self.it.append(self.g)

                # self.save_data()

                self.ok = False

            dis_steps = np.mean(self.distances) - dist_ant
            if np.max(self.distances) == previous_dist:
                break

            self.g += 1

        if (self.distances >= self.exploration_distance).any():
            done = True
            self.type_error = 'action_zone'
            self.calculate_error()
            print("MSE az:", self.dict_error)
            self.type_error = 'peaks'
            self.calculate_error()
            print("Error peak:", self.dict_error_peak)
            self.type_error = 'all_map'
            self.calculate_error()
            self.map_mse.append(self.error)
        else:
            done = False
        return done

    def error_value(self):
        return self.error_data

    def data_out(self):

        """
        Return the first and the last position of the particles (drones).
        """

        return self.X_test, self.secure, self.bench_function, self.grid_min, self.sigma, \
               self.mu, self.error_data, self.it, self.part_ant, self.bench_array, self.grid_or

    def return_bench(self):
        return self.centers_bench, self.dict_limits_bench, self.center_peaks_bench

    def print_error(self):
        print("Error peak:", np.mean(np.array(self.error_peak)), '+-', np.std(np.array(self.error_peak)) * 1.96)
        print("MSE action:", np.mean(np.array(self.action_mse)), '+-', np.std(np.array(self.action_mse)) * 1.96)
        print("MSE map:", np.mean(np.array(self.map_mse)), '+-', np.std(np.array(self.map_mse)) * 1.96)
        df1 = pd.DataFrame(self.error_peak)
        df1.to_excel('../Test/Results/Error/ErrorGridPath.xlsx')
        df2 = pd.DataFrame(self.action_mse)
        df2.to_excel('../Test/Results/MSEAZ/MSEAZGridPath.xlsx')
        df3 = pd.DataFrame(self.map_mse)
        df3.to_excel('../Test/Results/MSEM/MSEMGridPath.xlsx')

