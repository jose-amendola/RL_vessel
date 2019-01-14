import numpy as np
from pydyna_env import PyDynaEnv, VarVelPyDynaEnv
# from simulated_ship_env import SimulatedShipEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger
import pickle


import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

scenario = 'Aframax_Full_revTannuri_Cond1-intstep-2s.p3d'
# Get the environment and extract the number of actions.
file_name = scenario + '_' + timestamp
env = VarVelPyDynaEnv(p3d_name=scenario, report=True, report_name=file_name, n_steps=5)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


loaded_processor = None
with open('processor_Aframax_Full_revTannuri_Cond1-intstep-2s.p3d_20190107182538.pickle', 'rb') as f:
    loaded_processor = pickle.load(f)

with tf.device('/cpu:0'):
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
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=5,
                   target_model_update=1000, policy=policy, processor=loaded_processor)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    # dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format('ship_env'), overwrite=True)
    dqn.load_weights('C:\\Users\\jose_amendola\\RL_vessel\\dqn_varvel_Aframax_Full_revTannuri_Cond1-intstep-2s.p3d_8000000_20190107182538.h5f')
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=1, visualize=True)