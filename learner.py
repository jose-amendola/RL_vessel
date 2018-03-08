from sklearn.svm import SVR
import numpy as np


class Learner(object):
    def __init__(self):
        self.batch_list = list()
        self.learner = SVR(kernel='rbf', C=1e3, gamma=0.1)


    def add_to_batch(self, transition_list):
        self.batch_list = self.batch_list + transition_list

    def fit_batch(self, max_iterations):
        # samples = np.fromiter(map(lambda k: np.asarray(list(k[0])), self.batch_list), dtype = np.float64)
        states = [list(k[0]) for k in self.batch_list]
        actions = [ float(x[1]) for x in self.batch_list]
        rewards = np.asarray([ x[3] for x in self.batch_list], dtype=np.float64)
        q_target = rewards
        samples = np.column_stack((states, actions))
        for it in range(max_iterations):
            self.learner.fit(samples,q_target)
            q_prediction = np.fromiter(map(lambda state_action: self.learner.predict(state_action.reshape((1,7))), samples), dtype = np.float64)
            adj_q_prediction = np.append(q_prediction[1:], rewards[-1])
            q_target = rewards + adj_q_prediction