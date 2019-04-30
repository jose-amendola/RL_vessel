import numpy as np
from pydyna_env import PyDynaEnv, VarVelPyDynaEnv
# from simulated_ship_env import SimulatedShipEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from matplotlib import pyplot as plt

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

scenario = 'Aframax_Full_revTannuri_Cond1-intstep-2s.p3d'
# Get the environment and extract the number of actions.
file_name = scenario
env = PyDynaEnv(p3d_name=scenario, report=True, report_name=file_name, n_steps=5)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + (4,)))
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
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

reward_sequence = list()
iterations = list()
for i in range(1, 13):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    interval = 250000
    iterations.append(interval*i)
    file_name = 'dqn_'+scenario+'_{step}.h5f'
    file_name = file_name.format(step=interval*i)
    dqn.load_weights('C:\\Users\\jose_amendola\\RL_vessel\\'+file_name)
    dqn.test(env, nb_episodes=1, visualize=False)
    reward_array = [data[4] for data in env.transition_list]
    reward_sequence.append(np.sum(reward_array))

plt.plot(iterations, reward_sequence)
plt.xlabel('Training iterations')
plt.ylabel('Total reward per episode')
plt.grid(True)
plt.show()