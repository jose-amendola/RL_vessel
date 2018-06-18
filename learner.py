from sklearn.svm import SVR
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn import neighbors
import numpy as np
import actions
import pickle
import datetime
import reward
import datetime
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
import keras.models
from keras.utils import plot_model
from keras.callbacks import CSVLogger

import random

state_mode = 'simple_state'


def custom_metric(state_action_a, state_action_b):
    dist_list = list()
    weights = ((1/5000), (1/5000), (1/180), (1/5), (1/5), 1.0, 1.0)
    for vars in zip(state_action_a, state_action_b, weights):
        var_dist = np.float(abs((vars[0] - vars[1])*vars[2]))
        dist_list.append(var_dist)
    dist = np.average(dist_list)
    #TODO Use fixed divisors for ref distance..avoid zero
    return dist

def get_nn(obj):
    if obj:
        model = keras.models.load_model(obj)
    else:
        # create model
        model = Sequential()
        if state_mode == 'simple_state':
            model.add(Dense(20, input_shape=(4,), activation='relu'))
        else:
            model.add(Dense(20, input_shape=(7,), activation='relu'))
        model.add(Dense(20, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        # Compile model
        # sgd = optimizers.SGD(lr=1, decay=0, momentum=0.9, nesterov=True)
        # rmsprop = optimizers.RMSprop(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer='adam')
        # plot_model(model, to_file='model.png')
    return model




class Learner(object):

    exploring = None

    def __init__(self, file_to_save='agents/agent_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S'), load_saved_regression=False,
                 action_space_name='complete_angle',
                 r_m_=None, nn_=False):
        self.rw_mp = r_m_
        self.debug  = True
        self.current_step = 0
        self.batch_list = list()
        self.nn_flag = nn_
        if self.nn_flag:
            self.learner = get_nn(load_saved_regression)
        else:
            if load_saved_regression:
                self.learner = load_saved_regression
            else:
                # self.learner = SVR(kernel='rbf', C=1e3, gamma=0.1)
                # self.learner = RandomForestRegressor(n_estimators=20)
                # self.learner = tree.DecisionTreeRegressor(max_depth=4)
                self.learner = AdaBoostRegressor()
        self.end_states = dict()
        self.discount_factor = 0.8
        self.mode = 'angle_only'# self.mode = 'angle_and_rotation'#
        self.action_space = actions.BaseAction(action_space_name)
        self.states = list()
        self.act = list()
        self.rewards = list()
        self.states_p = list()
        self.q_target = list()
        r_mode = '__'
        self.file = None
        if self.rw_mp:
            r_mode = self.rw_mp.reward_mode
        self.file = file_to_save + self.learner.__class__.__name__ + '_r_' + r_mode+'_disc_'+str(self.discount_factor)
        self.logger = CSVLogger(self.file + 'log', separator=';', append=True)

    def add_to_batch(self, transition_list, final_flag):
        if self.rw_mp is not None:
            transition_list = replace_reward(transition_list)
        self.batch_list = self.batch_list + transition_list
        if final_flag != 0:
            self.end_states[len(self.batch_list)-1] = final_flag

    def add_tuples(self, tuples_list):
        self.batch_list = self.batch_list + tuples_list

    def load_sample_file(self, file_to_load):
        transitions = list()
        with open(file_to_load, 'rb') as infile:
            try:
                while True:
                    transitions = pickle.load(infile)
                    transitions = replace_reward(transitions)
            except EOFError as e:
                pass
        self.batch_list = self.batch_list + transitions


    def set_up_agent(self):
        print("Batch size: ", len(self.batch_list))
        self.states = [list(k[0]) for k in self.batch_list]
        if self.mode == 'angle_only':
            self.act = [(x[1][0]) for x in self.batch_list]
        else:
            self.act = [(x[1]) for x in self.batch_list]
        self.rewards = np.asarray([x[3] for x in self.batch_list], dtype=np.float64)
        self.states_p = [list(k[2]) for k in self.batch_list]
        self.end_states = [(x[4]) for x in self.batch_list]
        self.q_target = self.rewards
        # print('Debug final state :',self.q_target[-1])
        self.states = np.array(self.states)
        self.act = np.array(self.act)
        self.samples = np.column_stack((self.states, self.act))
        print("Current batch size: ", len(self.samples))
        # self.samples = self.normalize_state_action(self.samples)

    def fqi_step(self, max_iterations, debug=False):
        stop_flag = False
        for it in range(max_iterations):
            self.current_step += 1
            print("FQI_iteration: ", it)
            self.learner.fit(self.samples, self.q_target, batch_size=1000, verbose=1, nb_epoch=300, callbacks=[self.logger])
            # self.learner.fit(self.samples, self.q_target)
            if not self.nn_flag:
                sc = self.learner.score(self.samples, self.q_target)
                print("Score: ",sc)
            if self.discount_factor > 0:
                maxq_prediction = np.asarray([self.find_max_q(state_p)[0] for state_p in enumerate(self.states_p)])
                self.q_target = self.rewards + self.discount_factor*maxq_prediction
            else:
                self.q_target = self.rewards
            # print("Last rewards: ", self.rewards[-3:])
            # if (it % 1 == 0 and it != 0) or stop_flag:
            if self.nn_flag:
                self.learner.save(self.file+'it'+str(self.current_step)+'.h5')
            else:
                with open(self.file+'it'+str(it), 'wb') as outfile:
                    pickle.dump(self.learner, outfile)
            if stop_flag:
                break

    def find_max_q(self, state_p):
        qmax = -float('Inf')
        choice_list = list()
        if self.end_states[i] !=0:
            print('final')
            qmax = 0
        else:
            for action in self.action_space.action_combinations:
                if self.mode == 'angle_only':
                    state_action = np.append(state_p, action[0])
                else:
                    state_action = np.append(state_p, action)
                state_action = np.reshape(state_action, (1, -1))
                qpred = self.learner.predict(state_action, batch_size=1, verbose=1)[0][0]
                print('Q value and respective action: ', qpred, action)
                if qpred > qmax:
                    qmax = qpred
                    choice_list = [action]
                elif qpred == qmax:
                    choice_list.append(action)
        print('Returning Qmax value: ', qmax)
        return qmax, choice_list

    def select_action(self, state):
        qmax, choice_list = find_max_q(state)
        selected_action = random.choice(choice_list)
        print("Max",qmax)
        print(selected_action[0])
        print(selected_action[1])
        return selected_action

if __name__ == '__main__':
    with open('agent20180408200244', 'rb') as infile:
        agent_obj = pickle.load(infile)
        agent = Learner(load_saved_regression=agent_obj, action_space_name='only_rudder_action_space')
        action = agent.select_action((7977.5731952, 4594.6156251, -103.49968, -2.909885, -0.6988991, 0.0005474))

        print(action)