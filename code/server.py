# Server side
# ServerThread_Trace, use daemon thread

from numpy import array, linalg, diag, zeros

class server:

    def __init__(self, path, d, step_server, Lambda, P, S, T):
        self.path = path  # output path
        self.d = d # data dim
        self.step_server = step_server # step size of server
        self.Lambda = Lambda
        self.T = T  # number of tasks
        self.P = zeros((self.T, self.d))
        self.S = zeros((self.T, self.d))

    def run(self):
        grad = # received from tasks
        index =

        self.S[index] = grad # Change the corresponding row of the gradient matrix S with the new gradient vector sent from task
        u, s, vh = linalg.svd(self.P - self.step_server * self.S) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
        s[s < self.step_server * self.Lambda] = 0.0
        self.P = u.dot(diag(s)).dot(vh)

        # send corresponding column of P to task
        p_new = self.P[index]

