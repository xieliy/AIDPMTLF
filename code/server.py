# Server side

from numpy import linalg, diag, zeros

class server:

    def __init__(self, path, server_conn, lock, d, step_server, Lambda, T):
        self.path = path  # output path
        self.server_conn = server_conn  # connection object
        self.lock = lock
        self.d = d # data dim
        self.step_server = step_server # step size of server
        self.Lambda = Lambda
        self.T = T  # number of tasks
        self.P = zeros((self.T, self.d))
        self.S = zeros((self.T, self.d))

    def run(self):

        for i in xrange(self.T):
            self.server_conn.put((zeros(self.d), i)) # start from server, send init p_new to each task

        while not self.server_conn.empty():
            grad, index = self.server_conn.get() # received from tasks

            self.lock.acquire()
            self.S[index] = grad # Change the corresponding row of the gradient matrix S with the new gradient vector sent from task
            u, s, vh = linalg.svd(self.P - self.step_server * self.S) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
            s[s < self.step_server * self.Lambda] = 0.0
            self.P = u.dot(diag(s)).dot(vh)
            p_new = self.P[index] # send corresponding column of P to task
            self.lock.release()

            self.server_conn.put((p_new, index)) # send p_new and index to task

