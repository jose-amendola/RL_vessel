from sklearn.svm import SVR


class Learner(object):
    def __init__(self):
        self.batch_list = list()

    def add_to_batch(self, transition_list):
        self.batch_list = self.batch_list + transition_list

    def fit_batch(self, max_iterations):
        pass