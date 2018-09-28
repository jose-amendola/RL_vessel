import sys

#import pylab as plb
import numpy as np
import ship_env as ship


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self, state=None):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(-1, 2)

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        pass

# implement your own agent here
        
class QLearning():
    def __init__(self, *args, **kwargs):
        nd = 30
        nt = 60
        nv = 20
        ntp = 40
        self.s = np.zeros((nd+1, nt+1, nv+1, ntp+1, 4))
        for i in range(nd+1):
            for j in range(nt+1):
                for k in range(nv+1):
                    for l in range(ntp+1):
                        self.s[i,j,k,l,0] = 100*i/nd
                        self.s[i,j,k,l,1] = -180+180*j/nt
                        self.s[i,j,k,l,2] = 4*k/nv
                        self.s[i,j,k,l,3] = -1+2*l/ntp
        self.actions = [[-0.6,-0.4,-0.2,0,0.2,0.4,0.6],
                        [-0.6,-0.4,-0.2,0,0.2,0.4,0.6]]
        self.discount = kwargs.get("discount", 0.9)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.last_action = [0,0]
        self.Q = np.zeros((len(self.actions[0]), len(self.actions[1])))
        self.W = np.zeros((len(self.actions[0]), len(self.actions[1]), self.s.shape[0], self.s.shape[1], self.s.shape[2], self.s.shape[3]))
        self.phi = np.zeros((self.s.shape[0], self.s.shape[1], self.s.shape[2], self.s.shape[3]))
        self.epsilon = 0.2

    def update(self, next_state, reward, step):
        phi = np.exp(-(next_state[0]-self.s[:,:,:,:,0])**2)*np.exp(-(next_state[1]-self.s[:,:,:,:,1])**2)*np.exp(-(next_state[2]-self.s[:,:,:,:,2])**2)*np.exp(-(next_state[3]-self.s[:,:,:,:,3])**2)
        Q = np.sum(self.W*phi,axis=(2,3,4,5))
        difference = reward + self.discount*np.max(Q) - self.Q[self.last_action[0],self.last_action[1]]
        self.Q = Q[:]
        self.W[self.last_action[0],self.last_action[1],:,:,:,:] = self.W[self.last_action[0],self.last_action[1],:,:,:,:] + self.learning_rate*difference*self.phi
        self.phi = phi[:,:]
            
    def act(self, state=None, train = True):
        epsilon = self.epsilon
        if not train:
            phi = np.exp(-(state[0]-self.s[:,:,:,:,0])**2)*np.exp(-(state[1]-self.s[:,:,:,:,1])**2)*np.exp(-(state[2]-self.s[:,:,:,:,2])**2)*np.exp(-(state[3]-self.s[:,:,:,:,3])**2)
            self.Q = np.sum(self.W*phi,axis=(2,3,4,5))
            epsilon = 1
        if np.random.choice([False, True], p=[epsilon, 1-epsilon]):
            self.last_action = np.random.randint(0, 7, 2)
        else:
            self.last_action = np.unravel_index(np.argmax(self.Q),(len(self.actions[0]), len(self.actions[1])))
        return [-1,self.actions[1][self.last_action[1]]]
    
    def getW(self):
        return self.W
    
    def setEps(self,eps):
        self.epsilon=eps
    
class DQLearning(): #TODO consertar e validar (talvez)
    def __init__(self, *args, **kwargs):
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
        p = 300
        k = 80
        self.s = np.zeros((p+1, k+1, 2))
        for i in range(p+1):
            for j in range(k+1):
                self.s[i,j,0] = -150+150*i/p
                self.s[i,j,1] = -20+40*j/k
        self.actions = [-1, 0, 1]
        self.discount = kwargs.get("discount", 0.95)
        self.learning_rate = kwargs.get("learning_rate", 0.15)
        self.last_action = 0
        self.Q = [0, 0, 0]
        self.W = np.zeros((3, self.s.shape[0], self.s.shape[1]))
        self.phi = np.zeros((self.s.shape[0], self.s.shape[1]))

    def update(self, next_state, reward, step):
        phi = np.exp(-(next_state[0]-self.s[:,:,0])**2)*np.exp(-(next_state[1]-self.s[:,:,1])**2)
        Q = np.sum(self.W*phi,axis=(1,2))
        difference = reward + self.discount*np.max(Q) - self.Q[self.last_action]
        self.Q = Q[:]
        self.W[self.last_action,:,:] = self.W[self.last_action,:,:] + self.learning_rate*difference*self.phi
        self.phi = phi[:,:]
            
    def act(self, state=None):
        epsilon = 0.1
        if state != None:
            phi = np.exp(-(state[0]-self.s[:,:,0])**2)*np.exp(-(state[1]-self.s[:,:,1])**2)
            self.Q = np.sum(self.W*phi,axis=(1,2))
            epsilon = 1
        if np.random.choice([False, True], p=[epsilon, 1-epsilon]):
            self.last_action = np.random.randint(0, 3)
        else:
            self.last_action = np.argmax(self.Q)
        return self.actions[self.last_action]
    
class TD(): #TODO adaptar ao nosso algoritmo, se possível e nos interessar
    def __init__(self, *args, **kwargs):
        nd = 20
        nt = 60
        nv = 20
        ntp = 40
        self.s = np.zeros((nd+1, nt+1, nv+1, ntp+1, 4))
        self.e = 0
        for i in range(nd+1):
            for j in range(nt+1):
                for k in range(nv+1):
                    for l in range(ntp+1):
                        self.s[i,j,k,l,0] = 100*i/nd
                        self.s[i,j,k,l,1] = -180+180*j/nt
                        self.s[i,j,k,l,2] = 4*k/nv
                        self.s[i,j,k,l,3] = -1+2*l/ntp
        self.V_prime = 0
        self.W = np.zeros((self.s.shape[0], self.s.shape[1], self.s.shape[2], self.s.shape[3]))
        self.actions = [[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1],
                        [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]]
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.discount = kwargs.get('discount', 0.6)
        self.phi = np.zeros((self.s.shape[0], self.s.shape[1], self.s.shape[2], self.s.shape[3]))
        self.gamma = 0.1
        self.e = np.zeros((self.s.shape[0], self.s.shape[1], self.s.shape[2], self.s.shape[3]))
        self.last_action = [0,0]
        self.V_old = 0
        self.last_vx = 0

    def update(self, next_state, reward, step):
        if step==0:
            self.e = np.zeros((self.s.shape[0], self.s.shape[1], self.s.shape[2], self.s.shape[3]))
            self.V_old = 0
            self.V_prime = 0
            self.phi = np.exp(-(next_state[0]-self.s[:,:,:,:,0])**2)*np.exp(-(next_state[1]-self.s[:,:,:,:,1])**2)*np.exp(-(next_state[2]-self.s[:,:,:,:,2])**2)*np.exp(-(next_state[3]-self.s[:,:,:,:,3])**2)
        self.V_old = self.V_prime
        phi_prime = np.exp(-(next_state[0]-self.s[:,:,:,:,0])**2)*np.exp(-(next_state[1]-self.s[:,:,:,:,1])**2)*np.exp(-(next_state[2]-self.s[:,:,:,:,2])**2)*np.exp(-(next_state[3]-self.s[:,:,:,:,3])**2)
        V = np.sum(self.W*self.phi,axis=(0,1,2,3))
        self.V_prime = np.sum(self.W*phi_prime,axis=(0,1,2,3))
        self.e = self.discount*self.gamma*self.e + (1-self.learning_rate*self.discount*self.gamma*self.e*self.phi)*self.phi
        delta = reward + self.discount*self.V_prime - V
        self.W = self.W + self.learning_rate*(delta+V-self.V_old)*self.e - self.learning_rate*(V-self.V_old)*self.phi
        self.phi = phi_prime[:,:,:,:]
        self.last_vx = next_state[1]
        

    def act(self, state=None):
        if self.V_old < self.V_prime:
            self.last_action = np.sign(self.last_vx)
        elif self.V_old == self.V_prime:
            self.last_action == np.random.randint(-1, 2)
        else:
            self.last_action = -np.sign(self.last_vx)
        return self.last_action
            

# test class, you do not need to modify this class
class Tester:

    def __init__(self, agent):
        self.ship = ship.ShipEnv()
        self.agent = agent

    def visualize_trial(self, n_steps=400):
        """
        Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # make sure the ship is reset
        observation = self.ship.reset()
        negative_side = False
        if observation[0]<0:
            observation[0] = -observation[0]
            observation[1] = -180-observation[1]
            observation[3] = -observation[3]
            negative_side = True
        for n in range(n_steps):
#            print('\rt =', n)
#            print("Enter to continue...")
#            input()
#            sys.stdout.flush()
            self.ship.render()
            action = self.agent.act(observation, False)
            if negative_side:
                action[0] = -action[0]
            observation, reward, done, info = self.ship.step(action)
            negative_side = False
            if done:
                break
            if observation[0]<0:
                observation[0] = -observation[0]
                observation[1] = -180-observation[1]
                observation[3] = -observation[3]
                negative_side = True

#            # check for rewards
#            if reward > 0.0:
#                print("\rreward obtained at t = ", self.mountain_car.t)
#                break

    def learn(self, n_episodes, max_episode):
        """
        params:
            n_episodes: number of episodes to perform
            max_episode: maximum number of steps on one episode, 0 if unbounded
        """

        rewards = np.zeros(n_episodes)
        for c_episodes in range(1, n_episodes):
            observation = self.ship.reset()
            negative_side = False
            if observation[0]<0:
                observation[0] = -observation[0]
                observation[1] = -180-observation[1]
                observation[3] = -observation[3]
                negative_side = True
            step = 1
            print('EPISODE '+str(c_episodes))
            while step <= max_episode or max_episode <= 0:
                #print('EPISODE '+str(c_episodes))
                self.ship.render()
                action = self.agent.act()
                if negative_side:
                    action[0] = -action[0]
                observation, reward, done, info = self.ship.step(action)
                self.agent.update(observation, reward, step)
                rewards[c_episodes] += reward
                negative_side = False
                if done:
                    break
                if observation[0]<0:
                    observation[0] = -observation[0]
                    observation[1] = -180-observation[1]
                    observation[3] = -observation[3]
                    negative_side = True
                step += 1
            formating = "end of episode after {0:3.0f} steps,\
                           cumulative reward obtained: {1:1.2f}"
            print(formating.format(step-1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    agent = QLearning() #TD()
    test = Tester(agent)
            
    # modify RandomAgent by your own agent with the parameters you want
    # you can (and probably will) change these values, to make your system
    # learn longer
    n_steps = 10000
    test.learn(101, n_steps)
    print("End of learning, press Enter to visualize...")
    input()
    test.visualize_trial(n_steps)
