from gym import Env, spaces
import numpy as np
from environment import Environment
from utils import global_to_local
from shapely.geometry import LineString, Point
from viewer import Viewer


class ShipEnv(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        self.observation_space = spaces.Box(low=np.array([-100, -180, 0, -1.0]), high=np.array([100, 0, 4.0, 1.0]))
        self.start_pos = 15000.0
        self.buzz_interface = Environment()
        self.buzz_interface.set_up()
        self.point_a = (14000, 0)
        self.point_b = (15010,0)
        self.line = LineString([self.point_a, self.point_b])
        self.set_point = np.array([0, -90, 2.5, 0])
        self.tolerance = np.array([20, 2.0, 0.2, 0.05])
        self.last_pos = list()
        self.reset()
        self.plot = False
        self.viewer = None

    def step(self, action):
        info = dict()
        state_prime, _ = self.buzz_interface.step(angle_level=action[0], rot_level=action[1])
        v_lon, v_drift, _ = global_to_local(state_prime[3], state_prime[4], state_prime[2])
        obs = self.convert_state(state_prime)
        print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        rew = self.calculate_reward(obs=obs)
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        return obs, rew, dn, info

    def calculate_reward(self, obs):
        if not self.observation_space.contains(obs):
            return -1000
        elif np.any(np.abs(obs - self.set_point) > self.tolerance):
            return -0.1*(obs[0]**2)-0.1*((obs[2]-self.set_point[2])**2)-0.05*((obs[1]+90)**2)
        else:
            return 0

    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or -20000 > state_prime[0] or state_prime[0] > 20000 or -4000 > state_prime[1] or state_prime[1] > 4000:
            self.viewer.end_of_episode()
            return True
        else:
            return False

    #Agent handles the state space (distance_from_line,  heading, v_longitudinal,, heading_p)
    def convert_state(self, state):
        v_lon, v_drift, _ = global_to_local(state[3], state[4], state[2])
        ship_point = Point((state[0], state[1]))
        dist = ship_point.distance(self.line)
        if state[1] < self.point_a[1]: #It only works for lines parallel to x-axis
            dist = - dist
        obs = np.array([dist, state[2], v_lon, state[5]])
        return obs

    def reset(self):
        init = list(map(float, self.observation_space.sample()))
        self.buzz_interface.set_single_start_pos_mode([self.start_pos, init[0], init[1], init[2], 0, init[3]])
        self.buzz_interface.move_to_next_start()
        print('Reseting position')
        state = self.buzz_interface.get_state()
        self.last_pos = [state[0], state[1], state[2]]
        return self.convert_state(state)

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self.viewer = Viewer()
                self.viewer.plot_guidance_line(self.point_a, self.point_b)
            self.viewer.plot_position(self.last_pos[0], self.last_pos[1], self.last_pos[2])

    def close(self):
        pass

if __name__ == '__main__':
    mode = 'qlearning'
    if mode=='normal':
        env = ShipEnv()
        for i_episode in range(20):
            observation = env.reset()
            for t in range(10000):
                env.render()
                #print('State observed:', observation)
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
    elif mode=='qlearning':
        env = ShipEnv()
        #ds = (env.observation_space.high-env.observation_space.low)/np.array([25,45,10,10])
        ds = [4,4,0.4,0.2]
        #da = (env.action_space.high-env.action_space.low)/np.array([4,4])
        da = [0.2,0.2]
        s_dims = (env.observation_space.high-env.observation_space.low)/ds
        a_dims = (env.action_space.high-env.action_space.low)/da
        Q = np.zeros((26,46,11,11,11,11))
        n_episodes = 10000
        epsilon = 0.3
        alpha = 0.1
        gamma = 0.9
        for i_episode in range(n_episodes):
            observation = env.reset()
            for t in range(100000):
                print('EPISODE '+str(i_episode))
                env.render()
                if observation[0]<0:
                    observation[0] = -observation[0]
                    observation[1] = -180-observation[1]
                    observation[3] = -observation[3]
                obs_d = np.floor((observation - np.array([0, -180, 0, -1.0]))/ds)
                #print(obs_d)
                rand = np.random.uniform()
                if rand < epsilon:
                    act_d = np.random.randint(0, 11, 2).tolist()
                else:
                    act_d = np.unravel_index(np.argmax(Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3])]),(11,11))
                print(act_d)
                action = np.array(act_d)*da + env.action_space.low
                #action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if observation[0]<0:
                    observation[0] = -observation[0]
                    observation[1] = -180-observation[1]
                    observation[3] = -observation[3]
                new_obs_d = np.floor((np.array(observation) - np.array([0, -180, 0, -1.0]))/ds).tolist()
                Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3]),int(act_d[0]),int(act_d[1])] = (1-alpha)*Q[int(obs_d[0]),int(obs_d[1]),int(obs_d[2]),int(obs_d[3]),int(act_d[0]),int(act_d[1])] + alpha*(reward + gamma*np.amax(Q[int(new_obs_d[0]),int(new_obs_d[1]),int(new_obs_d[2]),int(new_obs_d[3])]))
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
                