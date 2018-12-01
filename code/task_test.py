
class task_test:
    def __init__(self, task_conn, index, ITER):
        self.task_conn = task_conn # connection object
        self.index = index # task number
        self.ITER = ITER # number of total iteration
        self.countITER = 0 # number of current iteration

    def run(self):
        while not self.task_conn.empty():
            msg, ind = self.task_conn.get()  # received from central server
            if self.index == ind:
                msg = msg + ind + 1
                self.task_conn.put((msg, ind)) # send grad and index to server
                self.countITER = self.countITER + 1
                if self.countITER == self.ITER:
                    break

