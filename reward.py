import numpy as np
import math
from simulation_settings import *


class RewardMapper(object):
    def __init__(self, r_mode_='cte'):
        self.ship = None
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

    def set_goal(self, point, heading, vel_l):
        self.goal_point = point
        self.g_vel_x, self.g_vel_y, self.g_heading_n_cw = utils.local_to_global(vel_l, 0, heading)

    def initialize_ship(self, x, y, heading, global_vel_x, global_vel_y, global_vel_theta):
        self.ship_last_vel = [global_vel_x, global_vel_y, global_vel_theta]
        self.ship_last_pos = [x, y, heading]
        self.ship_pos = self.ship_last_pos
        self.ship_vel = self.ship_last_vel
        self.ship = self.last_ship

    def update_ship(self, x, y, heading, global_vel_x, global_vel_y, global_vel_theta, angle, rot):
        self.ship_last_pos = self.ship_pos
        self.ship_last_vel = self.ship_vel
        self.last_ship = self.ship
        self.last_angle_selected = angle
        self.last_rot_selected = rot
        self.ship_vel = [global_vel_x, global_vel_y, global_vel_theta]
        self.ship_pos = [x, y, heading]


    def get_reward(self):
        # ref_array = np.array((self.goal_point[0], self.goal_point[1], self.g_heading_n_cw, self.g_vel_x, self.g_vel_y, 0))
        # array = np.array((self.ship_pos+self.ship_vel))
        ref_array = np.array((self.goal_point[0], self.goal_point[1], self.g_heading_n_cw))
        array = np.array((self.ship_pos))
        old_array = np.array((self.ship_last_pos))
        # new_dist = np.linalg.norm(array - ref_array)
        # old_dist = np.linalg.norm(old_array - ref_array)
        new_u_misalign = abs(array[2] - ref_array[2])
        old_u_misalign = abs(old_array[2] - ref_array[2])
        # print('distance_from_goal_state: ', new_dist)
        # shore_dist = self.boundary.exterior.distance(self.ship)
        # old_guid_dist = self.guid_line.distance(self.last_ship)
        # new_guid_dist = self.guid_line.distance(self.ship)
        # old_shore_dist = self.boundary.boundary.distance(self.last_ship)
        # new_shore_dist = self.boundary.boundary.distance(self.ship)
        new_u_balance = abs(geom_helper.get_shore_balance(array[0], array[1]))
        old_u_balance = abs(geom_helper.get_shore_balance(old_array[0], old_array[1]))
        # old_balance = self.get_shore_balance(old_array[0], old_array[1])
        # old_misalign = old_array[2] - ref_array[2]
        reward = -0.1
        if self.reward_mode == 'cte':
            reward = -0.1
        elif self.reward_mode == 'potential':
            pass
        elif self.reward_mode == 'rule':
            pass
            # if (old_balance < 0 and old_misalign > 0 and self.last_angle_selected != - 0.5) or \
            #         (old_balance > 0 and old_misalign < 0 and self.last_angle_selected != 0.5):
            #     reward = -100
            # elif (old_balance == 0 and old_misalign == 0 and self.last_angle_selected == - 0.5):
            #     reward = -100
            # else:
            #     reward = 100
        elif self.reward_mode == 'dist':
            reward = 100*math.exp(-0.000001*old_u_balance-0.000001*old_u_misalign)
        elif self.reward_mode == 'align':
            reward = 100*math.exp(-0.000001*old_u_misalign)
        elif self.reward_mode == 'step':
            if new_u_balance < 50 and new_u_misalign < 2:
                reward = 1
            else:
                reward = -0.1
        elif self.reward_mode == 'step_with_rudder_punish':
            if new_u_balance < 50 and new_u_misalign < 2:
                reward = 1
            else:
                reward = -0.1
            if abs(self.last_angle_selected) == 0.5:
                reward = reward - 0.2
        elif self.reward_mode == 'linear_with_rudder_punish':
            if new_u_balance < 30 and new_u_misalign < 2:
                reward = 1
            else:
                reward = -0.1 - 0.00001*new_u_balance
            if abs(self.last_angle_selected) == 0.5:
                reward = reward - 0.2
        geom_helper.set_polygon_position(array[0], array[1], array[2])
        viewer.plot_position(array[0], array[1], array[2])
        if geom_helper.ship_collided():
            print('SHIP COLLIDED!!!')
            reward = -1
            return reward
        return reward

if __name__ == "__main__":
    reward_map = RewardMapper()
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
    reward_map.set_shore_lines(upper_shore, lower_shore)
    bal = reward_map.get_shore_balance(8000, 4570)
    print(bal)
    print("Stop")
