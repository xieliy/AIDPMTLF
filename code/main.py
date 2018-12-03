import csv
from task import task
from server import server
from multiprocessing import Queue, Lock, Process
import numpy as np

# Parameters that need to adjust
dataset = 'synthetic' # 'real'
privacy = 0.005 # [0.005, 0.001, 0.05, 0.1]

path_project = '/home/xieliyan/Dropbox/GPU/GPU1/AIDPMTLF/' # path of project
path_data = path_project + "data/" # path of data
path_results = path_project + "results/" # path of results
T = 10 # number of task
wait_time = [i+1 for i in xrange(T)] # waiting time in one iteration, each task
p_train = 0.7 # percentage of training set
Lambda = 0.001 # regularization parameter
d = 28
ITER = 1000
p_ite = 200
step_task = 0.1
step_server = 0.01
mu = 0.0 # Gaussisan noise
alpha = 10.0
C1 = 1.0 # data is normalized to 1.0
C2 = 0.25 # bound on the derivavtive of logistic loss
data_num = 200 # number of data points in each local task
sigma = np.sqrt((2.0 * ITER * alpha * (step_server ** 2) * (C1 ** 2) * (C2 ** 2)) / (privacy * (data_num ** 2)))
data_all = [[] for i in xrange(T)] # data for all tasks
label_all = [[] for i in xrange(T)] # label for all tasks
tasks = [] # create tasks

lock = Lock()

conn = Queue() # creating a Queue between tasks and server
server_ins = server(path_results, conn, lock, d, step_server, Lambda, T, mu, sigma) # create server

if dataset == 'synthetic':
    for k in xrange(T):
        with open(path_data + 'data' + str(k + 1), 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                data_all[k].append(row[0].split(' '))
        data_all[k].pop(0) # remove first one

        N = len(data_all[k])
        for i in xrange(N):
            del data_all[k][i][-1]
            data_all[k][i] = [float(data_all[k][i][j]) for j in xrange(d)]

        with open(path_data + 'labelc' + str(k + 1), 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                label_all[k].append(float(row[0][:-1]))
        label_all[k].pop(0)

        tasks.append(task(path_results, dataset, data_all[k], label_all[k], conn, k, Lambda, ITER, p_ite, step_task, d, p_train, wait_time[k]))
        tasks[-1].model_init()
elif dataset == 'real':
    for k in xrange(T):
        with open(path_data + 'data' + str(k + 1), 'rb') as f:

            reader = csv.reader(f)
            for row in reader:
                data_all[k].append(row[0].split(' '))
        data_all[k].pop(0)  # remove first one

        N = len(data_all[k])
        for i in xrange(N):
            del data_all[k][i][-1]
            data_all[k][i] = [float(data_all[k][i][j])
                              for j in xrange(d)]

        with open(path_data + 'labelc' + str(k + 1), 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                label_all[k].append(float(row[0][:-1]))
        label_all[k].pop(0)

        tasks.append(task(path_results, dataset, data_all[k], label_all[k],
                          conn, k, Lambda, ITER,
                          p_ite, step_task, d, p_train,
                          wait_time[k]))
        tasks[-1].model_init()
else:
    raise Exception('Please choose the correct data sets!') # if choose incorrect data set

server_p = Process(target=server_ins.run()) # create server process
server_p.daemon = True # server_p will stop after main process stop
server_p.start() # start from server

tasks_p = [] # create tasks processes
for k in xrange(T):
    tasks_p.append(Process(target=tasks[k].run()))
    tasks_p[-1].start()