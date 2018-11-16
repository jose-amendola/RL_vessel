#!/usr/bin/python
#-*- coding: utf-8 -*-

from gym import Env, spaces
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from viewer_b import Viewer
from simulator import Simulator
from simulation_settings import buoys, goal, goal_factor, lower_shore, upper_shore
from shapely import affinity
from utils import global_to_local
import cProfile, pstats, io

# pr = cProfile.Profile()
class SimulatedShipEnv(Env):
    def __init__(self):
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
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([-1, -180, -1.0, 0.5]),
                                            high=np.array([1, -50, 1.0, 4.5]))
        self.init_space = spaces.Box(low=np.array([0, -np.pi/15, 1.0, 0.2,  -0.01]), high=np.array([30, np.pi/15, 2.0, 0.3, 0.01]))
        self.last_pos = np.zeros(3) # last_pos = [xg yg thg]
        self.last_action = np.zeros(1) #only one action
        self.simulator = Simulator()
        self.point_a = (13000, 0)
        self.point_b = (15010, 0)
        self.line = LineString([self.point_a, self.point_b])
        self.ship_point = None
        self.ship_polygon = Polygon(((-10, 0), (0, 100), (10, 0)))
        self.start_pos = list()
        self.viewer = None
        self.last_rew = 0
        self.symmetry = 1

    def step(self, action):
        rot_action = 0.6
        rudder_action, rot_action = self.convert_action(action)
        rudder_action *= self.symmetry
        print('Rudder lvl: ', rudder_action)
        print('Rot lvl: ', rot_action)
        state_prime = self.simulator.step(angle_level=rudder_action, rot_level=rot_action)
        # transforma variáveis do simulador em variáveis observáveis
        obs = self.convert_state_sog_cog(state_prime)
        print('Observed state: ', obs)
        if obs[0] < 0:
            obs[0], obs[1], obs[2] = - obs[0], -166.6-(obs[1]+166.6), -obs[2]
            self.symmetry = -1
            print('Inverted symmetry signal!!')
            print('Reflected obs: ', obs)
        else:
            self.symmetry = 1
        print('Action: ', action)
        dn = self.end(state_prime=state_prime, obs=obs)
        # rew = self.calculate_reward(obs=obs)
        rew = self.calculate_reward(obs)
        self.last_rew = rew
        if dn:
            if not self.goal_rec.contains(self.ship_point):
                rew -= 1000
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = self.convert_action(action)
        print('Reward: ', rew)
        info = dict()
        return obs, rew, dn, info

    def control_step(self, action):
        rot_action = 0.3
        rudder_action = action
        print('Rudder lvl: ', rudder_action)
        # pr.enable()
        state_prime = self.simulator.step(angle_level=rudder_action, rot_level=rot_action)
        # pr.disable()
        # transforma variáveis do simulador em variáveis observáveis
        obs = self.convert_state_sog_cog(state_prime)

        print('Action: ', action)
        print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        # rew = self.calculate_reward(obs=obs)
        rew = -1
        if dn:
            if not self.goal_rec.contains(self.ship_point):
                rew = -100000000
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = action
        print('Reward: ', rew)
        info = dict()
        return obs, rew, dn, info

    def convert_action(self, act):
        if act == 0:
            return -0.2, -0.2
        elif act == 1:
            return 0.0, -0.2
        elif act == 2:
            return 0.2, -0.2
        elif act == 3:
            return -0.2, 0.0
        elif act == 4:
            return 0.0, 0.0
        elif act == 5:
            return 0.2, 0.0
        elif act == 6:
            return -0.2, 0.6
        elif act == 7:
            return 0.0, 0.6
        elif act == 8:
            return 0.2, 0.6

    def convert_state_sog_cog(self, state):
        theta_deg = np.rad2deg(state[2])
        v_lon, v_drift, _ = global_to_local(state[3], state[4], theta_deg)
        self.ship_point = Point((state[0], state[1]))
        self.ship_polygon = affinity.translate(self.ship_polygon, state[0], state[1])
        self.ship_polygon = affinity.rotate(self.ship_polygon, -theta_deg, 'center')
        bank_balance = (self.ship_point.distance(self.upper_line) - self.ship_point.distance(self.lower_line)) / \
                       (self.ship_point.distance(self.upper_line) + self.ship_point.distance(self.lower_line))
        sog = np.linalg.norm([state[3], state[4]])
        cog = np.degrees(np.arctan2(state[4], state[3]))
        if cog > 0:
            cog -= 360
        obs = np.array([bank_balance, cog, state[5], sog])
        # print('Observation', obs)
        return obs

    def convert_state(self, state):
        """
        This method generated the features used to build the reward function
        :param state: Global state of the ship
        """
        ship_point = Point((state[0], state[1]))
        side = np.sign(state[1] - self.point_a[1])
        d = ship_point.distance(self.line)  # meters
        theta = side*state[2]  # radians
        vx = state[3]  # m/s
        vy = side*state[4]  # m/s
        thetadot = side * state[5]  # graus/min
        obs = np.array([d, theta, vx, vy, thetadot])
        return obs

    def calculate_reward(self, obs):
            rew = -np.abs(obs[1] + 166.6)**2 - 1000*np.abs(obs[2])**2 - np.abs(obs[3])**2
            return rew
        # if abs(obs[1]+166.6) < 0.3 and abs(obs[2]) < 0.01:
        #     return 1
        # else:
        #     return np.tanh(-((obs[1]+166.6)**2)-1000*(obs[2]**2))

    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or not self.boundary.contains(self.ship_point):
            print('Ending episode with obs: ', obs)
            if self.viewer is not None:
                self.viewer.end_episode()
            return True
        else:
            return False

    def set_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

    def reset(self):
        # init = list(map(float, self.init_space.sample()))
        theta = np.deg2rad(90-(-106.4))
        init = [11000, 5280, theta, 3*np.cos(theta), 3*np.sin(theta), 0]
        self.simulator.reset_start_pos(np.array(init))
        print('Reseting position')
        state = self.simulator.get_state()
        self.last_pos = np.array([state[0], state[1],  state[2]])
        return self.convert_state_sog_cog(state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.plot_boundary(self.buoys)
            # self.viewer.plot_guidance_line(self.point_a, self.point_b)
        self.viewer.plot_position(self.last_pos[0], self.last_pos[1],  self.last_pos[2],  20*self.last_action[0])

    def close(self, ):
        if self.viewer:
            self.viewer.freeze_screen()


if __name__ == '__main__':
    mode = 'normal'
    last_obs = np.array([0.00, 0.00, 0.00])
    if mode == 'normal':
        env = SimulatedShipEnv()
        for i_episode in range(1):
            last_obs = env.reset()
            for t in range(10):
                # env.render()
                action = - (last_obs[1]+166.6) - 20 * last_obs[2]
                last_obs, reward, done, info = env.control_step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())