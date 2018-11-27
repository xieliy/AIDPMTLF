# Task side

from numpy import dot, array, sign
import pickle

class task:

    def __init__(self, path, data, label, index, Lambda, ITER, p_ite, p_train):
        self.path = path  # output path
        self.index = index # task number
        self.Lambda = Lambda # regularization parameter
        self.ITER = ITER # number of current iteration
        self.p_ite = p_ite # number of iterations to print out
        self.data_train = data[:int(len(data) * p_train)]  # training data
        self.data_test = data[int(len(data) * p_train):]
        self.label_train = label[:int(len(data) * p_train)]  # training label
        self.label_test = label[int(len(data) * p_train):]
        self.countITER = 0 # number of current iteration
        self.error_all = [] # error rate of all iterations
        self.model = [] # model
        self.fn = 'error_all' + str(self.index) # file name

    def measurec(self, data, label, model):
        '''Classification Error rate of task model in each iteration'''
        pre = array(data).dot(array(model))
        b = sign(pre) == sign(label) # https://stackoverflow.com/questions/43380840/check-if-two-numeric-values-have-same-sign-in-numpy
        err = (len(b) - sum(b)) / (len(b) * 1.0) # error rate
        self.error_all.append(err)
        if self.ITER % self.p_ite == 0:
            print "This is task " + str(self.index) + ", the error rate in iteration "  + str(self.ITER) + " is " + str(err)

    def run(self):
        # Read q_{t}^{0} in task t, do only one time for each task, for whole project



    def store(self):
        pickle.dump(self.error_all, open(self.path + self.fn, 'w')) # store error rate of all iterations


if __name__ == '__name__':
