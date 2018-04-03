from sklearn.svm import SVR
from sklearn import tree
# from sklearn.ensemble import RandomForestRegressor
import numpy as np
import actions
import pickle
import datetime


class Learner(object):

    exploring = None

    def __init__(self, file_to_save='default_agent', load_saved_regression=False, action_space_name='simple_action_space'):
        self.batch_list = list()
        if load_saved_regression:
            self.learner = load_saved_regression
        else:
            # self.learner = SVR(kernel='rbf', C=1, gamma=0.1)
            # self.learner = RandomForestRegressor()
            self.learner = tree.DecisionTreeRegressor()
        self.end_states = list()
        self.file = file_to_save
        self.discount_factor = 1.0
        self.mode = 'angle_only'
        self.action_space = actions.BaseAction(action_space_name)

    def add_to_batch(self, transition_list, final_flag):
        if final_flag != 0:
            #TODO Fix not all end_states are being identified
            self.end_states.append(transition_list[-1][2])
        self.batch_list = self.batch_list + transition_list

    def fit_batch(self, max_iterations):
        states = [list(k[0]) for k in self.batch_list]
        if self.mode == 'angle_only':
            act = [(x[1][0]) for x in self.batch_list]
        rewards = np.asarray([x[3] for x in self.batch_list], dtype=np.float64)
        states_p = [list(k[2]) for k in self.batch_list]
        q_target = rewards
        states = np.array(states)
        act = np.array(act)
        samples = np.column_stack((states, act))
        # Using only rudder as action
        for it in range(max_iterations):
            print("FQI_iteration: ", it)
            self.learner.fit(samples, q_target)
            self.save_tree()
            maxq_prediction = np.fromiter(map(lambda state_p: self.find_max_q(state_p), states_p), dtype=np.float64)
            q_target = rewards + self.discount_factor*maxq_prediction


    def find_max_q(self, state_p):
        qmax = -float('Inf')
        if state_p in self.end_states:
            qmax = 0
        else:
            for action in self.action_space.action_combinations:
                if self.mode == 'angle_only':
                    state_action = np.append(state_p, action[0])
                state_action = np.reshape(state_action, (1, -1))
                qpred = self.learner.predict(state_action)
                if qpred > qmax:
                    qmax = qpred
        return qmax

    def select_action(self, state):
        selected_action = None
        qmax = -float('Inf')
        for action in self.action_space.action_combinations:
            if self.mode == 'angle_only':
                state_action = np.append(state, action[0])
            state_action = np.reshape(state_action, (1, -1))
            qpred = self.learner.predict(state_action)
            if qpred > qmax:
                qmax = qpred
                selected_action = action
        return selected_action

    def save_tree(self):
        if isinstance(self.learner, tree.DecisionTreeRegressor):
            tree.export_graphviz(self.learner, out_file='tree.dot')

    def __del__(self):
        with open(self.file, 'wb') as outfile:
            pickle.dump(self.learner, outfile)