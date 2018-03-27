import numpy as np
from itertools import product
from shapely.geometry import Polygon, LineString, Point
from shapely import affinity
from viewer import Viewer
from math import sin, cos, radians
import numpy as np
import math
import utils


class RewardMapper(object):
    def __init__(self, plot_flag=True):
        self.boundary = None
        self.goal_rec = None
        self.ship_polygon = None
        self.ship = None
        self.plot_flag = plot_flag
        if self.plot_flag:
            self.view = Viewer()
        self.set_ship_geometry(((0,0),(10,10),(0,20)))
        self.ship_vel = list()
        self.ship_pos = list()
        self.goal_point = None
        self.g_heading_n_cw = None
        self.g_vel_x = None
        self.g_vel_y = None

    def generate_inner_positions(self):
        points_dict = dict()
        for line_x in range(int(self.goal_point[0]+100), int(self.goal_point[0] + 6000), 500):
            line = LineString([(line_x, 0), (line_x, 15000)])
            intersect = self.boundary.intersection(line)
            if intersect.geom_type == 'LineString':
                middle_point = (line_x, (intersect.bounds[1]+intersect.bounds[3]) / 2)
                upper_point = (line_x, (middle_point[1] + intersect.bounds[3]) / 2)
                lower_point = (line_x, (middle_point[1] + intersect.bounds[1]) / 2)
                dist = Point(self.goal_point).distance(Point(middle_point))
                points_dict[middle_point] = dist
                points_dict[upper_point] = dist
                points_dict[lower_point] = dist
        return points_dict

    def set_boundary_points(self, points):
        self.boundary = Polygon(points)
        if self.plot_flag:
            self.view.plot_boundary(points)

    def set_ship_geometry(self, points):
        self.ship_polygon = Polygon(points)

    def set_goal(self, point, heading, vel_l):
        factor = 100
        self.goal_point = point
        self.g_vel_x, self.g_vel_y, self.g_heading_n_cw = utils.local_to_global(vel_l, 0, heading)
        self.goal_rec = Polygon(((point[0] - factor, point[1] - factor), (point[0] - factor, point[1] + factor), (point[0] + factor, point[1] + factor),
                            (point[0] + factor, point[1] - factor)))
        if self.plot_flag:
            self.view.plot_goal(point, factor)

    def update_ship(self, x, y, heading, global_vel_x, global_vel_y, global_vel_theta):
        self.ship_vel = [global_vel_x, global_vel_y, global_vel_theta]
        self.ship_pos = [x,y,heading]
        self.ship = affinity.translate(self.ship_polygon, x, y)
        self.ship = affinity.rotate(self.ship, heading, 'center')
        if self.plot_flag:
            self.view.plot_position(x, y, heading)

    def get_shortest_distance_from_boundary(self):
        a = self.ship.distance(self.boundary)
        return a

    def collided(self):
        collided = (not self.boundary.contains(self.ship))
        if collided:
            print('Collided!!')
        return collided

    def reached_goal(self):
        cont = self.goal_rec.contains(self.ship)
        reached = cont and abs(self.ship_vel[0]) < abs(self.g_vel_x) and abs(self.ship_pos[2] - self.g_heading_n_cw) < 20
        if reached:
            print('Reached goal!!')
        return reached

    def get_reward(self):
        ref_array = np.array((self.goal_point[0], self.goal_point[1], self.g_heading_n_cw, self.g_vel_x, self.g_vel_y, 0))
        array = np.array((self.ship_pos+self.ship_vel))
        dist = np.linalg.norm(array - ref_array)
        print('distance_from_goal_state: ', dist)
        # reward = -0.1*math.exp(-1/dist)
        reward = -0.1
        if self.collided():
            reward = -1
        goal = self.reached_goal()
        if goal:
            reward = 1
        return reward

if __name__ == "__main__":
    reward_map = RewardMapper(True)
    reward_map.update_ship(20, 20, 30, 0, 0, 0)
    print(reward_map.collided())
    # reward_map.update_ship_position(200, 200, 50)
    # reward_map.set_goal((100, 1000))
    for i in range(500):
        reward_map.update_ship(i, i, i, 0, 0, 0)
        ret = reward_map.collided()
        if ret:
            print(ret)

