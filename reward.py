import numpy as np
import math
from simulation_settings import *
import utils


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
        ref_array = np.array((self.goal_point[0], self.goal_point[1], self.g_heading_n_cw))
        array = np.array((self.ship_pos))
        vel_array = np.array((self.ship_vel[0], self.ship_vel[1]))
        ref_vel = np.array((self.g_vel_x, self.g_vel_y))
        # old_array = np.array((self.ship_last_pos))
        new_u_misalign = abs(array[2] - ref_array[2])
        # old_u_misalign = abs(old_array[2] - ref_array[2])
        new_u_balance = abs(geom_helper.get_shore_balance(array[0], array[1]))
        new_u_vel_diff = abs(np.linalg.norm(ref_vel - vel_array))
        # old_u_balance = abs(geom_helper.get_shore_balance(old_array[0], old_array[1]))
        reward = -0.1
        if self.reward_mode == 'quadratic':
            quadratic = -new_u_misalign**2
            reward += quadratic
        punish_rudder = -self.last_angle_selected**2
        reward += punish_rudder
        geom_helper.set_polygon_position(array[0], array[1], array[2])
        if geom_helper.ship_collided():
            print('SHIP COLLIDED!!!')
            reward = -1000000
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
