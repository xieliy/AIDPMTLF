from numpy import zeros

class server_test:

    def __init__(self, server_conn, lock, T):
        self.server_conn = server_conn  # connection object
        self.lock = lock
        self.T = T  # number of tasks

    def run(self):
        for i in xrange(self.T):
            self.server_conn.put((0, i)) # start from server, send init p_new to each task
        while not self.server_conn.empty():
            msg, ind = self.server_conn.get() # received from tasks
            self.lock.acquire()
            msg = msg + 0.5
            self.lock.release()
            self.server_conn.put((msg, ind)) # send p_new and index to task