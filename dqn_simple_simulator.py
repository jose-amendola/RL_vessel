import numpy as np
# from ship_env import ShipEnv
from pydyna_env import VarVelPyDynaEnv
from simulated_ship_env import SimulatedShipEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.callbacks import ModelIntervalCheckpoint
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import WhiteningNormalizerProcessor
import pickle


import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


processor = WhiteningNormalizerProcessor()
with open('processor_Aframax_Full_revTannuri_Cond1-intstep-2s.p3d_20190107182538.pickle', 'rb') as f:
    processor = pickle.load(f)



scenario = 'Aframax_Full_revTannuri_Cond1-intstep-2s.p3d'
# Get the environment and extract the number of actions.
env = VarVelPyDynaEnv(p3d_name=scenario, report=False, n_steps=5)
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
# policy = EpsGreedyQPolicy(eps=0.2)
# policy = BoltzmannQPolicy()
policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.2, value_test=.05,
                              nb_steps=1000000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1,
               target_model_update=1000, policy=policy, processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('C:\\Users\\jose_amendola\\RL_vessel\\dqn_ship_env_varvel_dist_20190107182538_Aframax_Full_revTannuri_Cond1-intstep-2s.p3d_weights.h5f')

dqn.fit(env, nb_steps=10e6, visualize=False, verbose=1, action_repetition=1, log_interval=100000,
        callbacks=[ModelIntervalCheckpoint(filepath='dqn_varvel_'+scenario+'_{step}_'+timestamp+'.h5f', interval=100000)])
dqn.save_weights('dqn_{}_{}_{}_weights.h5f'.format('ship_env_varvel_dist', timestamp, scenario), overwrite=True)

env.report = True
env.report_name = file_name = scenario + '_' + timestamp
# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=1, visualize=True)

with open('processor_'+scenario+'_'+timestamp+'.pickle', 'wb') as outfile:
    pickle.dump(processor, outfile)