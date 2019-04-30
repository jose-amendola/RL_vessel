from gym import Env, spaces
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from pydyna import FastTimeWrapper
from simulation_settings import buoys, goal, goal_factor, lower_shore, upper_shore
from shapely import affinity
from utils import global_to_local
from viewer_b import Viewer
import inspect
import csv
import datetime

class PyDynaEnv(Env):
    def __init__(self, p3d_name='Aframax_Full_revTannuri_Cond1-intstep-2s.p3d', report=True, report_name=None, n_steps=5):
        self.buoys = buoys
        self.lower_shore = lower_shore
        self.upper_shore = upper_shore
        self.goal = goal
        self.goal_factor = goal_factor
        self.upper_line = LineString(upper_shore)
        self.lower_line = LineString(lower_shore)
        self.goal_point = Point(goal)
        self.boundary = Polygon(self.buoys)
        self.goal_rec = Polygon(((self.goal[0] - self.goal_factor, self.goal[1] - self.goal_factor),
                                 (self.goal[0] - self.goal_factor, self.goal[1] + self.goal_factor),
                                 (self.goal[0] + self.goal_factor, self.goal[1] + self.goal_factor),
                                 (self.goal[0] + self.goal_factor, self.goal[1] - self.goal_factor)))
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.array([-1.0, -185, -1.0, -1.0]),
                                            high=np.array([1.0, -150, 1.0, 1.0]))
        self.last_pos = np.zeros(5) # last_pos = [xg yg thg]
        self.last_action = np.zeros(2)
        self.simulator = FastTimeWrapper()
        self.simulator.load_p3d(p3d_name)
        self.simulator.select_vessel('36')
        self.point_a = (13000, 0)
        self.point_b = (15010, 0)
        self.line = LineString([self.point_a, self.point_b])
        self.ship_point = None
        self.ship_polygon = LineString(((0, 0), (0, 100)))
        self.start_pos = list()
        self.viewer = None
        self.last_rew = 0
        self.symmetry = 1
        self.original_state = self.simulator.get_vessel_pos_vel()
        self.n_steps = n_steps
        self.report = report
        self.transition_list = list()
        self.metrics = list()
        self.report_name = report_name

    def step(self, action):
        rudder_action, rot_action = self.convert_action(action)
        # rudder_action *= self.symmetry
        # print('Rudder lvl: ', rudder_action)
        # print('Rot lvl: ', rot_action)
        self.simulator.set_norm_dem_rudder_level(rudder_action)
        self.simulator.set_norm_dem_prop_level(rot_action)
        for _ in range(self.n_steps):
            self.simulator.advance()
        state_prime = self.simulator.get_vessel_pos_vel()
        # print('original state:', state_prime)
        # transforma variáveis do simulador em variáveis observáveis
        obs = self.convert_state_sog_cog(state_prime)
        # print('Observed state: ', obs)
        # if obs[0] < 0:
        #     obs[0], obs[1], obs[2] = - obs[0], -166.6-(obs[1]+166.6), -obs[2]
        #     self.symmetry = -1
        #     print('Inverted symmetry signal!!')
        #     print('Reflected obs: ', obs)
        # else:
        #     self.symmetry = 1
        # print('Action: ', action)
        theta = self.angle_from_dyna(state_prime[5])
        # self.ship_polygon = affinity.translate(self.ship_polygon, state_prime[0]-self.last_pos[0], state_prime[1]-self.last_pos[1])
        # self.ship_polygon = affinity.rotate(self.ship_polygon, -(theta-self.last_pos[2]), 'center')
        dn = self.end(state_prime=state_prime, obs=obs)
        # rew = self.calculate_reward(obs=obs)
        rew = self.calculate_reward(obs)
        self.last_rew = rew
        # if dn:
        #     if not self.goal_rec.contains(self.ship_point):
        #         rew -= 1000000*obs[3]
        if dn:
            if not self.goal_rec.contains(self.ship_point) or obs[3] > 2.0:
                rew -= 10
            if self.report:
                self.dump_report()
        self.last_pos = [state_prime[0], state_prime[1], theta]
        self.last_action = [rudder_action, rot_action]
        # print('Reward: ', rew)
        info = dict()
        if self.report:
            self.save_transition(state_prime, obs, rew, rudder_action)
        return obs, rew, dn, info

    def calculate_reward(self, obs):
        # rew = 1-np.abs(obs[1] + 166.6)
        rew = (1 + obs[2] * (obs[1] + 166.6)) / (1 + np.abs(obs[0])) / (1 + np.abs(obs[1] + 166.6))
        return rew

    def convert_state_sog_cog(self, state):
        theta_deg = self.angle_from_dyna(state[5])
        self.ship_point = Point((state[0], state[1]))
        bank_balance = (self.ship_point.distance(self.upper_line) - self.ship_point.distance(self.lower_line)) / \
                       (self.ship_point.distance(self.upper_line) + self.ship_point.distance(self.lower_line))
        # sog = np.linalg.norm([state[6], state[7]])
        cog = np.degrees(np.arctan2(state[7], state[6]))
        # goal_dist = self.goal_point.distance(self.ship_point)
        # drift_angle = -self.angle_from_dyna(state[5])-270.0 - cog
        if cog > 0:
            cog -= 360
        rate_of_turn = -np.rad2deg(state[11])
        obs = np.array([bank_balance, cog, rate_of_turn, self.last_action[0]])
        # print('Observation', obs)
        return obs

    def save_transition(self, org_state, converted_state, reward, action):
        data = (org_state[0], org_state[1], self.angle_from_dyna(org_state[5]), converted_state[0], reward, action)
        self.transition_list.append(data)

    def dump_report(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        with open('data\\trajectory_' + self.report_name + '_' + timestamp+'.csv', 'wt') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['x', 'y', 'heading', 'norma_lat_dev', 'reward', 'action'])
            for tr in self.transition_list:
                csv_out.writerow(tr)
        with open('data\\stats_' + self.report_name + '_' + timestamp+'.csv', 'wt') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['Average lateral deviation', 'Variance', 'Average heading'])
            nld_array = [data[3] for data in self.transition_list]
            heading_array = [data[2] for data in self.transition_list]
            reward_array = [data[4] for data in self.transition_list]
            avg_nld = np.average(nld_array)
            var_nld = np.var(nld_array)
            avg_heading = np.average(heading_array)
            total_reward = np.sum(reward_array)
            csv_out.writerow([avg_nld, var_nld, avg_heading, total_reward])



    def angle_from_dyna(self, ang):
        ret = 90 - np.rad2deg(ang)
        while ret > 180.0:
            ret -= 360.0
        while ret < -180.0:
            ret -= 360.0
        return ret

    def convert_action(self, act):
        if act == 0:
            return -0.5, 0.6
        elif act == 1:
            return -0.2, 0.6
        elif act == 2:
            return 0.0, 0.6
        if act == 3:
            return 0.2, 0.6
        elif act == 4:
            return 0.5, 0.6



    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or not self.boundary.contains(self.ship_point):
            print('Ending episode with obs: ', obs)
            if self.viewer is not None:
                self.viewer.end_episode()
            return True
        else:
            return False

    def reset(self):
        init_pos = [11000, 5280]
        # init_pos = [9984, 4980]
        theta = np.deg2rad(90-(-103.4))
        init_vel = 3
        init = [init_pos[0], init_pos[1], self.original_state[2], self.original_state[3], self.original_state[4], theta,
                init_vel*np.cos(theta), init_vel*np.sin(theta), self.original_state[8],
                self.original_state[9], self.original_state[10], self.original_state[11]]
        self.simulator.reset_vessel_state(init)
        print('Reseting position')
        state = self.simulator.get_vessel_pos_vel()
        self.last_pos = np.array([state[0], state[1], self.angle_from_dyna(state[5])])
        self.ship_polygon = affinity.translate(self.ship_polygon, state[0],
                                               state[1])
        self.ship_polygon = affinity.rotate(self.ship_polygon, -self.angle_from_dyna(state[5]), 'center')
        self.last_action = np.zeros(2)
        self.transition_list = list()
        return self.convert_state_sog_cog(state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.plot_boundary(self.buoys)
            self.viewer.plot_goal(goal, goal_factor)
            # self.viewer.plot_guidance_line(self.point_a, self.point_b)
        self.viewer.plot_position(self.last_pos[0], self.last_pos[1],  self.last_pos[2],  20*self.last_action[0])

    def close(self, ):
        if self.viewer:
            self.viewer.freeze_screen()


class VarVelPyDynaEnv(Env):
    def __init__(self, p3d_name='Aframax_Full_revTannuri_Cond1-intstep-2s.p3d', report=True, report_name=None, n_steps=5):
        self.buoys = buoys
        self.lower_shore = lower_shore
        self.upper_shore = upper_shore
        self.goal = goal
        self.goal_factor = goal_factor
        self.upper_line = LineString(upper_shore)
        self.lower_line = LineString(lower_shore)
        self.goal_point = Point(goal)
        self.boundary = Polygon(self.buoys)
        self.goal_rec = Polygon(((self.goal[0] - self.goal_factor, self.goal[1] - self.goal_factor),
                                 (self.goal[0] - self.goal_factor, self.goal[1] + self.goal_factor),
                                 (self.goal[0] + self.goal_factor, self.goal[1] + self.goal_factor),
                                 (self.goal[0] + self.goal_factor, self.goal[1] - self.goal_factor)))
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=np.array([-1.0, -180, -1.0, -1.0, 1.0]),
                                            high=np.array([1.0, -150, 1.0, 1.0, 5.0]))
        self.last_pos = np.zeros(5) # last_pos = [xg yg thg]
        self.last_action = np.zeros(2)
        self.simulator = FastTimeWrapper()
        self.simulator.load_p3d(p3d_name)
        self.simulator.select_vessel('36')
        self.point_a = (13000, 0)
        self.point_b = (15010, 0)
        self.line = LineString([self.point_a, self.point_b])
        self.ship_point = None
        self.ship_polygon = LineString(((0, 0), (0, 100)))
        self.start_pos = list()
        self.viewer = None
        self.last_rew = 0
        self.symmetry = 1
        self.original_state = self.simulator.get_vessel_pos_vel()
        self.transition_list = list()
        self.n_steps = n_steps
        self.report = report
        self.report_name = report_name

    def step(self, action):
        rudder_action, rot_action = self.convert_action(action)
        # rudder_action *= self.symmetry
        print('Rudder lvl: ', rudder_action)
        print('Rot lvl: ', rot_action)
        self.simulator.set_norm_dem_rudder_level(rudder_action)
        self.simulator.set_norm_dem_prop_level(rot_action)
        for _ in range(self.n_steps):
            self.simulator.advance()
        state_prime = self.simulator.get_vessel_pos_vel()
        # print('original state:', state_prime)
        # transforma variáveis do simulador em variáveis observáveis
        obs = self.convert_state_sog_cog(state_prime)
        # print('Observed state: ', obs)
        # if obs[0] < 0:
        #     obs[0], obs[1], obs[2] = - obs[0], -166.6-(obs[1]+166.6), -obs[2]
        #     self.symmetry = -1
        #     print('Inverted symmetry signal!!')
        #     print('Reflected obs: ', obs)
        # else:
        #     self.symmetry = 1
        # print('Action: ', action)
        theta = self.angle_from_dyna(state_prime[5])
        # self.ship_polygon = affinity.translate(self.ship_polygon, state_prime[0]-self.last_pos[0], state_prime[1]-self.last_pos[1])
        # self.ship_polygon = affinity.rotate(self.ship_polygon, -(theta-self.last_pos[2]), 'center')
        dn = self.end(state_prime=state_prime, obs=obs)
        # rew = self.calculate_reward(obs=obs)
        rew = self.calculate_reward(obs)
        self.last_rew = rew
        # if dn:
        #     if not self.goal_rec.contains(self.ship_point):
        #         rew -= 1000000*obs[3]
        if dn:
            if not self.goal_rec.contains(self.ship_point) or obs[3] > 2.0:
                rew -= 10
            if self.report:
                self.dump_report()
        self.last_pos = [state_prime[0], state_prime[1], theta]
        self.last_action = [rudder_action, rot_action]
        print('Reward: ', rew)
        info = dict()
        if self.report:
            self.save_transition(state_prime, obs, rew, rudder_action)
        return obs, rew, dn, info


    def calculate_reward(self, obs):
        # rew = 1-np.abs(obs[1] + 166.6)
        rew = (1 + obs[2] * (obs[1] + 166.6)) / (1 + np.abs(obs[0])) / (1 + np.abs(obs[1] + 166.6)) / (1 + np.abs(obs[4] - 2.0))
        return rew

    def convert_state_sog_cog(self, state):
        theta_deg = self.angle_from_dyna(state[5])
        self.ship_point = Point((state[0], state[1]))
        bank_balance = (self.ship_point.distance(self.upper_line) - self.ship_point.distance(self.lower_line)) / \
                       (self.ship_point.distance(self.upper_line) + self.ship_point.distance(self.lower_line))
        sog = np.linalg.norm([state[6], state[7]])
        cog = np.degrees(np.arctan2(state[7], state[6]))
        # goal_dist = self.goal_point.distance(self.ship_point)
        # drift_angle = -self.angle_from_dyna(state[5])-270.0 - cog
        if cog > 0:
            cog -= 360
        rate_of_turn = -np.rad2deg(state[11])
        obs = np.array([bank_balance, cog, rate_of_turn, self.last_action[0], sog])
        # print('Observation', obs)
        return obs

    def save_transition(self, org_state, converted_state, reward, action):
        data = (org_state[0], org_state[1], self.angle_from_dyna(org_state[5]), converted_state[0],
                reward, action, converted_state[4])
        self.transition_list.append(data)

    def dump_report(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        with open('data\\trajectory_' + self.report_name + '_' + timestamp+'.csv', 'wt') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['x', 'y', 'heading', 'norma_lat_dev', 'reward', 'action', 'vel'])
            for tr in self.transition_list:
                csv_out.writerow(tr)
        with open('data\\stats_' + self.report_name + '_' + timestamp+'.csv', 'wt') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['Average lateral deviation', 'Variance', 'Average heading'])
            nld_array = [data[3] for data in self.transition_list]
            heading_array = [data[2] for data in self.transition_list]
            reward_array = [data[4] for data in self.transition_list]
            avg_nld = np.average(nld_array)
            var_nld = np.var(nld_array)
            avg_heading = np.average(heading_array)
            total_reward = np.sum(reward_array)
            csv_out.writerow([avg_nld, var_nld, avg_heading, total_reward])

    def angle_from_dyna(self, ang):
        ret = 90 - np.rad2deg(ang)
        while ret > 180.0:
            ret -= 360.0
        while ret < -180.0:
            ret -= 360.0
        return ret

    def convert_action(self, act):
        if act == 0:
            return -0.5, 0.6
        elif act == 1:
            return -0.2, 0.6
        elif act == 2:
            return 0.0, 0.6
        elif act == 3:
            return 0.2, 0.6
        elif act == 4:
            return 0.5, 0.6
        elif act == 5:
            return -0.5, 0.0
        elif act == 6:
            return -0.2, 0.0
        elif act == 7:
            return 0.0, 0.0
        elif act == 8:
            return 0.2, 0.0
        elif act == 9:
            return 0.5, 0.0

    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or not self.boundary.contains(self.ship_point):
            print('Ending episode with obs: ', obs)
            if self.viewer is not None:
                self.viewer.end_episode()
            return True
        else:
            return False

    def reset(self):
        init_pos = [11000, 5280]
        # init_pos = [9984, 4980]
        theta = np.deg2rad(90-(-103.4))
        init_vel = 3
        init = [init_pos[0], init_pos[1], self.original_state[2], self.original_state[3], self.original_state[4], theta,
                init_vel*np.cos(theta), init_vel*np.sin(theta), self.original_state[8],
                self.original_state[9], self.original_state[10], self.original_state[11]]
        self.simulator.reset_vessel_state(init)
        print('Reseting position')
        state = self.simulator.get_vessel_pos_vel()
        self.last_pos = np.array([state[0], state[1], self.angle_from_dyna(state[5])])
        self.ship_polygon = affinity.translate(self.ship_polygon, state[0],
                                               state[1])
        self.ship_polygon = affinity.rotate(self.ship_polygon, -self.angle_from_dyna(state[5]), 'center')
        self.last_action = np.zeros(2)
        return self.convert_state_sog_cog(state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.plot_boundary(self.buoys)
            self.viewer.plot_goal(goal, goal_factor)
            # self.viewer.plot_guidance_line(self.point_a, self.point_b)
        self.viewer.plot_position(self.last_pos[0], self.last_pos[1],  self.last_pos[2],  20*self.last_action[0])

    def close(self, ):
        if self.viewer:
            self.viewer.freeze_screen()
if __name__ == '__main__':
    env = PyDynaEnv()
    print(inspect.getsourcelines(env.convert_state_sog_cog))
    for ep in range(5):
        last_obs = env.reset()
        for _ in range(1000):
            env.render()
            last_obs = env.step(0)
            print('Obs:', last_obs)
            if last_obs[2] is True:
                break
