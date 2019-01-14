from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.memory import SequentialMemory
import numpy as np
import datetime
from pydyna_env import PyDynaEnv, VarVelPyDynaEnv


class ReusePolicy(EpsGreedyQPolicy):
    def __init__(self, agent_weights=None, *args, **kwargs):
        super(ReusePolicy, self).__init__(*args, **kwargs)
        self.nb_actions = 3
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + (4,)))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))
        self.memory = SequentialMemory(limit=500, window_length=1)
        self.policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=self.memory, nb_steps_warmup=5,
                       target_model_update=1e-2, policy=self.policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        self.dqn.load_weights(agent_weights)

    def simplify_obs(self, obs):
        simple_obs = [obs[0][0:4]]
        return simple_obs

    def select_action(self, q_values, state):
        assert q_values.ndim == 1
        nb_actions_current_env = q_values.shape[0]
        rand_number = np.random.uniform()
        if rand_number > self.eps:
            simple_obs = self.simplify_obs(state)
            reuse_q_values = self.dqn.compute_q_values(simple_obs)
            action = np.argmax(reuse_q_values)
        elif rand_number < (1 - self.eps)/2:
            action = np.argmax(q_values)
        else:
            action = np.random.random_integers(0, nb_actions_current_env - 1)
        return action


class ReuseDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(ReuseDQNAgent, self).__init__(*args, **kwargs)

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values, state=state)
        else:
            action = self.test_policy.select_action(q_values=q_values)
        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action
        return action


timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


# Get the environment and extract the number of actions.
env = VarVelPyDynaEnv()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

print(model.summary())


memory = SequentialMemory(limit=500, window_length=1)
policy = ReusePolicy(eps=0.5, agent_weights='dqn_ship_env_vel_goal_dist_20181214110654_weights.h5f')
dqn = ReuseDQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1,
               target_model_update=1000, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1, action_repetition=1, log_interval=100000)
dqn.save_weights('dqn_{}_{}_weights.h5f'.format('ship_env_vel_goal_dist', timestamp), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)