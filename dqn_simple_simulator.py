import numpy as np
# from ship_env import ShipEnv
from simulated_ship_env import SimulatedShipEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory


import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


# Get the environment and extract the number of actions.
env = SimulatedShipEnv()
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


# model = Sequential()
# tmp = (1,) + env.observation_space.shape
#
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500, window_length=1)
policy = EpsGreedyQPolicy(eps=0.2)
# policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=5,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('dqn_ship_env_vel_goal_dist_20181121093253_weights.h5f')
# dqn.load_weights('omae\\dqn_ship_env_20181106101409_init_105.4-ctespeed0.6-rw-col1000000000-np.abs(obs[1] + 166.6)2 - 1000np.abs(obs[2])2.h5f')
# dqn.load_weights('C:\\Users\\jose_amendola\\RL_vessel\\omae\\dqn_ship_env_weights_straight_line_tanh(-((obs[1]+103.4)2)-100(obs[2]2))_collision1000_nospeed.h5f')
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# for i in range(10):
#     dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
#
#     After training is done, we save the final weights.
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1, action_repetition=1)
dqn.save_weights('dqn_{}_{}_weights.h5f'.format('ship_env_vel_goal_dist', timestamp), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)