from gym import Env, spaces
import numpy as np
from environment import Environment
from utils import global_to_local
from shapely.geometry import LineString, Point, Polygon
from shapely import affinity
from viewer import Viewer
from simulation_settings import buoys, goal, goal_factor, lower_shore, upper_shore


class ShipEnv(Env):
    def __init__(self, special_mode = None):
        self.special_mode = special_mode
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
        if self.special_mode == 'fixed_rotation':
            self.action_space = spaces.Box(low=np.array([-0.5]), high=np.array([0.5]))
        else:
            self.action_space = spaces.Box(low=np.array([-0.5, -0.5]), high=np.array([0.5, 0.5]))
        self.observation_space = spaces.Box(low=np.array([-1, -180, 1.0, -2.0, -1.0]),
                                            high=np.array([1, -50, 4.0, 2.0, 1.0]))
        self.init_space = spaces.Box(low=np.array([-1, -103.3, 2.4]), high=np.array([1, -103.5, 2.6]))
        self.start_pos = [11000, 5300.10098]
        self.buzz_interface = Environment()
        self.buzz_interface.set_up()
        self.point_a = (13000, 0)
        self.point_b = (15010, 0)
        self.line = LineString([self.point_a, self.point_b])
        self.set_point = np.array([0, -103, 2.5, 0, 0])
        self.tolerance = np.array([20, 2.0, 0.2, 0.05])
        self.last_pos = list()
        self.ship_point = None
        self.ship_polygon = Polygon(((-10, 0), (0, 100), (10, 0)))
        self.reset()
        self.plot = False
        self.viewer = None
        self.last_action = [0,0]

    def step(self, action):
        info = dict()
        if self.special_mode == 'fixed_rotation':
            state_prime, _ = self.buzz_interface.step(angle_level=action[0], rot_level=0.3)
        else:
            state_prime, _ = self.buzz_interface.step(angle_level=action[0], rot_level=action[1])
        v_lon, v_drift, _ = global_to_local(state_prime[3], state_prime[4], state_prime[2])
        obs = self.convert_state(state_prime)
        # print('Action: ', action)
        # print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        if dn:
            if not self.goal_rec.contains(self.ship_point):
                rew = -1000
        else:
            rew = self.calculate_reward(obs=obs)
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = action
        return obs, rew, dn, info

    def calculate_reward(self, obs):
        if abs(obs[0]) < 0.02 and abs(obs[4]) < 0.0001 and abs(obs[1]+103.4) < 0.5:
            return 1000
        else:
            return np.tanh(-(obs[0]**2)-((obs[0]+103.4)**2)-((obs[2]-2.5)**2)-(obs[4]**2))

    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or not self.boundary.contains(self.ship_point):
            print('Ending episode with obs: ', obs)
            if self.viewer is not None:
                self.viewer.end_of_episode()
            return True
        else:
            return False

    #Agent handles the state space (distance_from_line, heading, v_longitudinal, heading_p)
    #obs variables:
    # 0:bank_balance
    # 1:heading
    # 2:v_lon
    # 3:v_drift
    # 4:heading_p
    def convert_state(self, state):
        # print('Original state:', state)
        v_lon, v_drift, _ = global_to_local(state[3], state[4], state[2])
        self.ship_point = Point((state[0], state[1]))
        self.ship_polygon = affinity.translate(self.ship_polygon,  state[0],  state[1])
        self.ship_polygon = affinity.rotate(self.ship_polygon, -state[2], 'center')
        bank_balance = (self.ship_point.distance(self.upper_line) - self.ship_point.distance(self.lower_line)) / \
                       (self.ship_point.distance(self.upper_line) + self.ship_point.distance(self.lower_line))
        obs = np.array([bank_balance, state[2], v_lon, v_drift, state[5]])
        # print('Observation', obs)
        return obs
    
    def change_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

    def reset(self):
        init = list(map(float, self.init_space.sample()))
        self.buzz_interface.set_single_start_pos_mode([self.start_pos[0], self.start_pos[1]+init[0], init[1], init[2], 0, 0])
        self.buzz_interface.move_to_next_start()
        print('Reseting position')
        state = self.buzz_interface.get_state()
        self.last_pos = [state[0], state[1], state[2]]
        self.last_action = [0,0]
        return self.convert_state(state)

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self.viewer = Viewer()
                # self.viewer.plot_guidance_line(self.point_a, self.point_b)
                self.viewer.plot_boundary(self.buoys)
                self.viewer.plot_goal(self.goal, self.goal_factor)
            self.viewer.plot_position(self.last_pos[0], self.last_pos[1], self.last_pos[2], 30*self.last_action[0])

    def close(self):
        self.buzz_interface.finish()

if __name__ == '__main__':
    mode = 'normal'
    if mode=='normal':
        env = ShipEnv()
        for i_episode in range(20):
            observation = env.reset()
            for t in range(10000):
                env.render()
                #print('State observed:', observation)
                # action = env.action_space.sample()
                action_rudder = np.clip((observation[0]-20*observation[4]), -1, 1)
                action = np.array([action_rudder, 0.3])
                observation, reward, done, info = env.step(action)
                print('action', action)
                print('Obs', observation)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
    elif mode=='qlearning':
        env = ShipEnv()
        #ds = (env.observation_space.high-env.observation_space.low)/np.array([25,45,10,10])
        ds = [5,4,0.4,0.2]
        #da = (env.action_space.high-env.action_space.low)/np.array([4,4])
        da = [0.25,0.25]
        s_dims = (env.observation_space.high-env.observation_space.low)/ds
        a_dims = (env.action_space.high-env.action_space.low)/da
        Q = np.zeros((21,31,11,11,9,9))
        n_episodes = 10000
        epsilon = 0.3
        alpha = 0.1
        gamma = 0.9
        for i_episode in range(n_episodes):
            observation = env.reset()
            negative_side = False
            if observation[0]<0:
                observation[0] = -observation[0]
                observation[1] = -180-observation[1]
                observation[3] = -observation[3]
                negative_side = True
            for t in range(100000):
                print('EPISODE '+str(i_episode))
                env.render()
                obs_d = np.floor((observation - np.array([0, -150, 0, -1.0]))/ds)
                #print(obs_d)
                rand = np.random.uniform()
                if rand < epsilon:
                    act_d = np.random.randint(0, 9, 2).tolist()
                else:
                    act_d = np.unravel_index(np.argmax(Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3])]),(9,9))
                print(act_d)
                action = np.array(act_d)*da + env.action_space.low
                if negative_side:
                    action[0] = -action[0]
                #action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                negative_side = False
                if observation[0]<0:
                    observation[0] = -observation[0]
                    observation[1] = -180-observation[1]
                    observation[3] = -observation[3]
                    negative_side = True
                new_obs_d = np.floor((np.array(observation) - np.array([0, -150, 0, -1.0]))/ds).tolist()
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3]),int(act_d[0]),int(act_d[1])] = (1-alpha)*Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3]),int(act_d[0]),int(act_d[1])] + alpha*reward
                    break
                else:
                    Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3]),int(act_d[0]),int(act_d[1])] = (1-alpha)*Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3]),int(act_d[0]),int(act_d[1])] + alpha*(reward + gamma*np.amax(Q[int(new_obs_d[0]),int(new_obs_d[1]),int(new_obs_d[2]),int(new_obs_d[3])]))
                