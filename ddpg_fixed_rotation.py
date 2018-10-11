from ship_env import ShipEnv
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
# from rl.callbacks import FileLogger, TrainEpisodeLogger

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


class Processor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)

# logger = FileLogger('C:\\Users\\jose_amendola\\RL_vessel\\agents\\log_'+timestamp, interval=5)
# logger = TrainEpisodeLogger()
# Get the environment and extract the number of actions.
env = ShipEnv(special_mode='fixed_rotation')
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(20))
actor.add(Activation('relu'))
actor.add(Dense(20))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(20)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(20)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('tanh')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3, processor=Processor())
agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
# agent.load_weights('ddpg_20181006160521_Ship_Env_weights.h5f')

steps = 20
for i in range(10):
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
    agent.fit(env, nb_steps=steps, visualize=False, verbose=1, log_interval=5)

    # After training is done, we save the final weights.
    agent.save_weights('ddpg_{}_{}_()_()_weights.h5f'.format(timestamp, 'Ship_Env', str(steps), str(i)), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=20000)