from sklearn.svm import SVR
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
import numpy as np
import actions
import pickle
import datetime
import reward
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

import types
import tempfile
import keras.models





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
        model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='rbf'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class Learner(object):

    exploring = None

    def __init__(self, file_to_save='default_agent', load_saved_regression=False,
                 action_space_name='simple_action_space',
                 r_m_=None, nn_=False):
        self.rw_mp = r_m_
        self.batch_list = list()
        self.nn_flag = nn_
        if self.nn_flag:
            self.learner = self.learner = get_nn(load_saved_regression)
        else:
            if load_saved_regression:
                self.learner = load_saved_regression
            else:
                pass
                # self.learner = neighbors.KNeighborsRegressor(2, weights='distance', metric=custom_metric)

                # self.nn_flag = True
                # self.learner = SVR(kernel='rbf', C=1e3, gamma=0.1)
                # self.learner = RandomForestRegressor()
                # self.learner = tree.DecisionTreeRegressor()
        self.end_states = dict()
        r_mode = '__'
        if self.rw_mp:
            r_mode = self.rw_mp.reward_mode
        self.file = file_to_save+self.learner.__class__.__name__+'_r_'+ r_mode
        self.discount_factor = 1.0
        self.mode = 'angle_and_rotation'# self.mode = 'angle_only'
        self.action_space = actions.BaseAction(action_space_name)
        self.states = list()
        self.act = list()
        self.rewards = list()
        self.states_p = list()
        self.q_target = list()
        # self.debug_file = open('debug_fqi'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.txt','w')

    def replace_reward(self, transition_list):
        new_list = list()
        for transition in transition_list:
            resulting_state = transition[2]
            action_selected = transition[1]
            self.rw_mp.update_ship(resulting_state[0], resulting_state[1], resulting_state[2], resulting_state[3],
                                   resulting_state[4], resulting_state[5], action_selected[0], action_selected[1])
            new_reward = self.rw_mp.get_reward()
            tmp = list(transition)
            tmp[3] = new_reward
            transition = tuple(tmp)
            new_list.append(transition)
        print(new_list[-1])
        return new_list

    def add_to_batch(self, transition_list, final_flag):
        if self.rw_mp is not None:
            transition_list = self.replace_reward(transition_list)
        self.batch_list = self.batch_list + transition_list
        if final_flag != 0:
            self.end_states[len(self.batch_list)-1] = final_flag

    def load_sample_file(self, file_to_load):
        transitions = list()
        with open(file_to_load, 'rb') as infile:
            try:
                while True:
                    transitions = pickle.load(infile)
                    transitions = self.replace_reward(transitions)
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
        self.states = np.array(self.states)
        self.act = np.array(self.act)
        self.samples = np.column_stack((self.states, self.act))

    def fqi_step(self, max_iterations, debug=False):
        for it in range(max_iterations):
            print("FQI_iteration: ", it)
            self.learner.fit(self.samples, self.q_target)
            maxq_prediction = np.asarray([self.find_max_q(i, state_p) for i,state_p in enumerate(self.states_p)])
            self.q_target = self.rewards + self.discount_factor*maxq_prediction
            if it % 1 == 0 and it != 0:
                if self.nn_flag:
                    self.learner.save(self.file)
                else:
                    with open(self.file, 'wb') as outfile:
                        pickle.dump(self.learner, outfile)
            # if debug:
                # print(self.q_target,file=self.debug_file)
                # print('\n\n', file=self.debug_file)



    def find_max_q(self, i, state_p):
        print('  >>>Finding max_q for : ',i)
        qmax = -float('Inf')
        # final = self.end_states.get(i)
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
                qpred = self.learner.predict(state_action)
                if qpred > qmax:
                    qmax = qpred
        return qmax

    def select_action(self, state):
        selected_action = None
        qmax = -float('Inf')
        print('Select action')
        for action in self.action_space.action_combinations:
            if self.mode == 'angle_only':
                state_action = np.append(state, action[0])
            else:
                state_action = np.append(state, action)
            state_action = np.reshape(state_action, (1, -1))
            qpred = self.learner.predict(state_action)
            if qpred > qmax:
                qmax = qpred
                selected_action = action
            #TODO Implement random choice for equal q value cases
        print(qmax)
        print(selected_action[0])
        print(selected_action[1])
        return selected_action

    def __del__(self):
        pass
        # self.debug_file.close()
        # with open(self.file, 'wb') as outfile:
        #     pickle.dump(self.learner, outfile)

if __name__ == '__main__':
    with open('agent20180408200244', 'rb') as infile:
        agent_obj = pickle.load(infile)
        agent = Learner(load_saved_regression=agent_obj, action_space_name='only_rudder_action_space')
        action = agent.select_action((7977.5731952, 4594.6156251, -103.49968, -2.909885, -0.6988991, 0.0005474))

        print(action)