# Task side

from numpy import dot, array, sign, zeros, mean
from math import log, exp
from numpy.linalg import norm
from scipy.optimize import minimize
import pickle
import time

class task:

    def __init__(self, path, data, label, task_conn, index, Lambda, ITER, p_ite, step_task, d, p_train, wait_time):
        self.path = path  # output path
        self.task_conn = task_conn # connection object
        self.index = index # task number
        self.Lambda = Lambda # regularization parameter
        self.ITER = ITER # number of total iteration
        self.p_ite = p_ite # number of iterations to print out
        self.step_task = step_task  # step size of task
        self.d = d # data dim
        self.wait_time = wait_time  # output path
        self.data_train = data[:int(len(data) * p_train)]  # training data
        self.data_test = data[int(len(data) * p_train):]
        self.label_train = label[:int(len(data) * p_train)]  # training label
        self.label_test = label[int(len(data) * p_train):]
        self.countITER = 0 # number of current iteration
        self.error_all = [] # error rate of all iterations
        self.q = [] # task specific component
        self.p = [] # shared component
        self.L = len(self.label_train) # number of training data
        self.fn = 'error_all' + str(self.index) # file name

    def measurec(self, data, label, model):
        '''Classification Error rate of task model in each iteration'''
        pre = array(data).dot(array(model))
        b = sign(pre) == sign(label) # https://stackoverflow.com/questions/43380840/check-if-two-numeric-values-have-same-sign-in-numpy
        err = (len(b) - sum(b)) / (len(b) * 1.0) # error rate
        self.error_all.append(err)
        if self.countITER % self.p_ite == 0:
            print "This is task " + str(self.index) + ", the error rate in iteration "  + str(self.countITER) + " is " + str(err)

    def lr(self, z):
        '''logistic loss'''
        logr = log(1.0 + exp(-z))
        return logr

    def obj(self, x):
        '''objective function of classification task, x here is q, in the first iteration, p=0'''
        jfd = self.lr(self.label_train[0] * dot(self.data_train[0], x))
        for i in xrange(1, self.L):
            jfd = jfd + self.lr(self.label_train[i] * dot(self.data_train[i], x))
        f = (1.0 / self.L) * jfd + (self.Lambda / 2.0) * (norm(x) ** 2)
        return f

    def model_init(self):
        '''$\q_{t}^{(0)}$ using STL'''
        x0 = zeros(self.d)
        self.q = minimize(self.obj, x0, method='Nelder-Mead').x  # minimization procedure
        self.p = zeros(self.d)

    def logistic_grad_q(self):
        '''grad of logistic loss w.r.t q'''
        v = [] # store values
        w = array(self.q) + array(self.p)
        for i in xrange(self.L):
            nyx = -1.0 * self.label_train[i] * array(self.data_train[i])
            v.append(nyx * exp(dot(nyx,w)) / (1.0 + exp(dot(nyx,w))))
        q_new = w - self.step_task * (mean(array(v), axis=0) + self.Lambda * array(self.q))
        return q_new

    def logistic_grad_p(self):
        '''grad of logistic loss w.r.t p'''
        v = [] # store values
        w = array(self.q) + array(self.p)
        for i in xrange(self.L):
            nyx = -1.0 * self.label_train[i] * array(self.data_train[i])
            v.append(nyx * exp(dot(nyx,w)) / (1.0 + exp(dot(nyx,w))))
        return mean(array(v), axis=0)

    def run(self):

        start_time = time.time()

        while not self.task_conn.empty():
            p_new, ind = self.task_conn.get()  # received from central server

            if self.index == ind:

                time.sleep(self.wait_time)  # sleep for a while.

                q_old = self.q
                self.q = self.logistic_grad_q()
                self.measurec(self.data_test, self.label_test, array(p_new) + array(q_old))
                grad = self.logistic_grad_p()

                self.task_conn.put((grad, self.index)) # send grad and index to server

                self.countITER = self.countITER + 1
                if self.countITER == self.ITER:
                    break

        duration = time.time() - start_time

        with open(self.path + 'duration.txt', 'a+') as f:  # store the training time
            f.write('Task' + str(self.index) + ':' +str(duration))
            f.write('\n')

        pickle.dump(self.error_all, open(self.path + self.fn, 'w'))  # store error rate of all iterations
        pickle.dump(array(self.q) + array(self.p), open(self.path + self.fn, 'w'))  # store error rate of all iterations