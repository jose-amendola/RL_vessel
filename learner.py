from sklearn.svm import SVR
import numpy as np
import actions


class Learner(object):
    def __init__(self):
        self.batch_list = list()
        self.learner = SVR(kernel='rbf', C=1, gamma=0.1)
        self.end_states = list()


    def add_to_batch(self, transition_list):
        self.end_states.append(transition_list[-1][2])
        self.batch_list = self.batch_list + transition_list

    def fit_batch(self, max_iterations):
        states = [list(k[0]) for k in self.batch_list]
        act = [float(x[1]) for x in self.batch_list]
        rewards = np.asarray([x[3] for x in self.batch_list], dtype=np.float64)
        states_p = [list(k[2]) for k in self.batch_list]
        q_target = rewards
        samples = np.column_stack((states, act))
        for it in range(max_iterations):
            self.learner.fit(samples, q_target)
            maxq_prediction = np.fromiter(map(lambda state_p: self.find_max_q(state_p), states_p), dtype=np.float64)

            q_target = rewards + maxq_prediction

    def find_max_q(self, state_p):
        qmax = -10000000000000000000
        if state_p in self.end_states:
            qmax = 0
        else:
            for action in actions.possible_actions:
                state_action = np.append(state,action)
                qpred = self.learner.predict(state_action)
                if qpred > qmax:
                    qmax = qpred
        return qmax