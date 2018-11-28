# Server side
# ServerThread_Trace, use daemon thread

from numpy import array

class server:

    def __init__(self, path, d, step_server, mu, P, S):
        self.path = path  # output path
        self.d = d # data dim
        self.step_server = step_server # step size of server
        self.mu = mu  #
        self.P = P
        self.S = S

    def run(self):
        grad = # received from tasks
        index =

        input = array(self.P) - self.step_server * array(self.S)




