import numpy as np
from itertools import product
from shapely.geometry import Polygon, LineString, Point
from shapely import affinity
from viewer import Viewer
from math import sin, cos, radians
import numpy as np
import math
import utils
from simulation_settings import *


class RewardMapper(object):
    def __init__(self, plot_flag=True, r_mode_='cte'):
        self.boundary = None
        self.goal_rec = None
        self.ship_polygon = None
        self.ship = None
        self.plot_flag = plot_flag
        if self.plot_flag:
            self.view = Viewer()
        self.set_ship_geometry(((0, 0), (0, 10), (20, 0)))
        self.ship_vel = list()
        self.ship_last_vel = list()
        self.ship_pos = list()
        self.ship_last_pos = list()
        self.last_ship = None
        self.goal_point = None
        self.g_heading_n_cw = None
        self.g_vel_x = None
        self.g_vel_y = None
        self.reward_mode = r_mode_
        self.last_angle_selected = None
        self.last_rot_selected = None
        self.guid_line = None
        self.upper_shore = None
        self.lower_shore = None

    def is_inbound_coordinate(self, x, y):
        return self.boundary.buffer(-20).contains(Point(x, y))

    def generate_inner_positions(self):
        points_dict = dict()
        for line_x in range(int(self.goal_point[0]+5000), int(self.goal_point[0] + 6000), 500):
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

    def get_middle_y(self, x):
        line = LineString([(x, 0), (x, 15000)])
        intersect = self.boundary.intersection(line)
        return (intersect.bounds[1] + intersect.bounds[3]) / 2

    def get_guidance_line(self):
        y_temp_a = self.get_middle_y(8000)
        y_temp_b = self.get_middle_y(9000)
        x_a = 3600
        x_b = 14000
        m,b = np.polyfit([8000,9000],[y_temp_a,y_temp_b],1)
        y_a = m*x_a+b
        y_b = m*x_b+b
        self.guid_line = LineString([(x_a, y_a), (x_b, y_b)])
        return (x_a, y_a), (x_b, y_b)

    def set_boundary_points(self, points):
        self.boundary = Polygon(points)
        if self.plot_flag:
            self.view.plot_boundary(points)

    def set_shore_lines(self, upper_points, lower_points):
        self.upper_shore = LineString(upper_points)
        self.lower_shore = LineString(lower_points)

    def get_shore_balance(self, x, y):
        ship_point = Point((x,y))
        #upper means positive sign
        upper_dist = ship_point.distance(self.upper_shore)
        lower_dist = ship_point.distance(self.lower_shore)
        return (upper_dist - lower_dist)



    def set_ship_geometry(self, points):
        self.ship_polygon = Polygon(points)

    def set_goal(self, point, heading, vel_l):
        factor = 300
        self.goal_point = point
        self.g_vel_x, self.g_vel_y, self.g_heading_n_cw = utils.local_to_global(vel_l, 0, heading)
        self.goal_rec = Polygon(((point[0] - factor, point[1] - factor), (point[0] - factor, point[1] + factor), (point[0] + factor, point[1] + factor),
                            (point[0] + factor, point[1] - factor)))
        if self.plot_flag:
            self.view.plot_goal(point, factor)

    def initialize_ship(self, x, y, heading, global_vel_x, global_vel_y, global_vel_theta):
        self.ship_last_vel = [global_vel_x, global_vel_y, global_vel_theta]
        self.ship_last_pos = [x, y, heading]
        self.last_ship = affinity.translate(self.ship_polygon, x, y)
        self.last_ship = affinity.rotate(self.last_ship, heading, 'center')
        self.ship_pos = self.ship_last_pos
        self.ship_vel = self.ship_last_vel
        self.ship = self.last_ship
        if self.plot_flag:
            self.view.plot_position(x, y, heading)

    def update_ship(self, x, y, heading, global_vel_x, global_vel_y, global_vel_theta, angle, rot):
        self.ship_last_pos = self.ship_pos
        self.ship_last_vel = self.ship_vel
        self.last_ship = self.ship
        self.last_angle_selected = angle
        self.last_rot_selected = rot
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
        # reached = cont and abs(self.ship_vel[0]) < abs(self.g_vel_x) and abs(self.ship_pos[2] - self.g_heading_n_cw) < 20
        # reached = abs(self.ship_pos[2] - self.g_heading_n_cw) < 20 and cont
        reached = cont
        # reached = abs(self.ship_vel[0]) < 0.2 or cont
        if reached:
            print('Reached goal!!')
        return reached

    def get_reward(self):
        # ref_array = np.array((self.goal_point[0], self.goal_point[1], self.g_heading_n_cw, self.g_vel_x, self.g_vel_y, 0))
        # array = np.array((self.ship_pos+self.ship_vel))
        ref_array = np.array((self.goal_point[0], self.goal_point[1], self.g_heading_n_cw))
        array = np.array((self.ship_pos))
        old_array = np.array((self.ship_last_pos))
        # new_dist = np.linalg.norm(array - ref_array)
        # old_dist = np.linalg.norm(old_array - ref_array)
        new_misalign = abs(array[2] - ref_array[2])
        old_misalign = abs(old_array[2] - ref_array[2])
        # print('distance_from_goal_state: ', new_dist)
        # shore_dist = self.boundary.exterior.distance(self.ship)
        old_guid_dist = self.guid_line.distance(self.last_ship)
        new_guid_dist = self.guid_line.distance(self.ship)
        old_shore_dist = self.boundary.boundary.distance(self.last_ship)
        new_shore_dist = self.boundary.boundary.distance(self.ship)
        reward = -0.1
        if self.reward_mode == 'cte':
            reward = -0.1
        elif self.reward_mode == 'potential':
        # #goal point field
        #     pot_goal = (1/(1+new_dist)) - (1/(1+old_dist))
        #     k_goal = 0
        # #guidance_field
        #     pot_guid = (1/(1+new_guid_dist)) - (1/(1+old_guid_dist))
        #     k_guid = 0
        # #alignment
        #     pot_align = (1/(1+new_align)) - (1/(1+old_align))
        #     k_align = 0.1
        # # collision repulsion field
            pot_collision = new_shore_dist - old_shore_dist
            k_collision = 0.1
        #     # reward = k_guid*pot_guid + k_align*pot_align
        #     if new_guid_dist < 10 and abs(new_align) < 3:
        #         reward = 100
        #     else:
        #         reward = -0.1*new_align - new_guid_dist
        #     pot_collision = new_shore_dist - old_shore_dist
        #     k_collision = 0.1
        #     reward = k_collision*pot_collision
        elif self.reward_mode == 'punish_align':
            if new_misalign < 2 and new_guid_dist < 1:
                reward = 100
            else:
                reward = -(new_misalign**2)
        if self.collided():
            reward = -100
            return reward
        goal = self.reached_goal() #
        if goal:
            reward = 0
        return reward

if __name__ == "__main__":
    reward_map = RewardMapper(False)
    # reward_map.update_ship(20, 20, 30, 0, 0, 0, 0, )
    # print(reward_map.collided())
    # # reward_map.update_ship_position(200, 200, 50)
    # # reward_map.set_goal((100, 1000))
    # for i in range(500):
    #     reward_map.update_ship(i, i, i, 0, 0, 0, 0, )
    #     ret = reward_map.collided()
    #     if ret:
    #         print(ret)
    reward_map.set_boundary_points(buoys)
    reward_map.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    reward_map.get_guidance_line()
    print("Stop")
