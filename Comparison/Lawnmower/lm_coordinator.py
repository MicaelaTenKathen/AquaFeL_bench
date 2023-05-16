import numpy as np
from shapely.geometry import Polygon
from Comparison.Lawnmower.grid_based_sweep_coverage_path_planner import planning, SweepSearcher


class Coordinator():
    def __init__(self, acq, map_data):
        self.acq_method = acq
        self.map_data = map_data
        _x, _y = self.obtain_shapely_polygon().exterior.coords.xy
        self.fpx, self.fpy = self.obtain_points(_x, _y)

    def obtain_shapely_polygon(self):
        reg = Polygon(
            [(0, 0), (0, np.size(self.map_data, 0)), (np.size(self.map_data, 1), np.size(self.map_data, 0)),
             (np.size(self.map_data, 1), 0)])
        return reg

    def obtain_points(self, x, y):
        if self.acq_method == "LD":
            bx, by = planning(x.tolist(), y.tolist(), 10 / 3, moving_direction=SweepSearcher.MovingDirection.LEFT,
                              sweeping_direction=SweepSearcher.SweepDirection.DOWN)
        if self.acq_method == "LU":
            bx, by = planning(x.tolist(), y.tolist(), 10 / 3, moving_direction=SweepSearcher.MovingDirection.LEFT,
                              sweeping_direction=SweepSearcher.SweepDirection.UP)
        if self.acq_method == "RD":
            bx, by = planning(x.tolist(), y.tolist(), 10 / 3, moving_direction=SweepSearcher.MovingDirection.RIGHT,
                              sweeping_direction=SweepSearcher.SweepDirection.DOWN)
        if self.acq_method == "RU":
            bx, by = planning(x.tolist(), y.tolist(), 10 / 3, moving_direction=SweepSearcher.MovingDirection.RIGHT,
                              sweeping_direction=SweepSearcher.SweepDirection.UP)
        px = []
        py = []
        for ipx, ipy in zip(bx, by):
            iipx = np.round(ipx).astype(np.int)
            iipy = np.round(ipy).astype(np.int)

            if self.map_data[iipy, iipx] == 1:
                px.append(iipx)
                py.append(iipy)
        # px = [px[i] for i in range(len(px)) if i % 3 == 0]
        # py = [py[i] for i in range(len(py)) if i % 3 == 0]
        print(px, py)
        return px, py

    def generate_new_goal(self):

        new_pos = np.array([self.fpx.pop(0), self.fpy.pop(0)])
        new_pos = np.append(new_pos, 0)
        # print('future pos is: ', new_pos)

        return new_pos
