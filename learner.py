from sklearn.svm import SVR
import numpy as np


class Learner(object):
    def __init__(self):
        self.batch_list = list()
        self.learner = SVR(kernel='rbf', C=1e3, gamma=0.1)
        self.alpha = 0.1

    def add_to_batch(self, transition_list):
        self.batch_list = self.batch_list + transition_list

    def fit_batch(self, max_iterations):
        pass
        # for it in range(max_iterations):
        #     for transition in self.batch_list:
        #         state = (transition[0]
        #         reward = transition[3]
        #         if it == 0:
        #             qprime
        #         q_target = q_current + self.alpha*(q_prime + reward - q_current)
        #         self.learner.fit(state)
