import csv
from task import task
from server import server
from multiprocessing import Queue, Process

path_project = '/home/xieliyan/Dropbox/GPU/GPU1/AIDPMTLF/' # path of project
path_data = path_project + "data/" # path of data
path_results = path_project + "results/" # path of results
T = 2 # number of task
p_train = 0.3 # percentage of training set
Lambda = 0.001 # regularization parameter
d = 28
ITER = 50000
p_ite = 1000
step_task = 0.1
step_server = 0.01
data_all = [[] for i in xrange(T)] # data for all tasks
label_all = [[] for i in xrange(T)] # label for all tasks
tasks = [] # create tasks

task_conn, server_conn = Queue()  # creating a Queue between tasks and server
server_ins = server(path_results, server_conn, d, step_server, Lambda, T) # create server
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

    tasks.append(task(path_results, data_all[k], label_all[k], task_conn, k, Lambda, ITER, p_ite, step_task, d, p_train))
    tasks[-1].model_init()

server_p = Process(target=server_ins.run()) # create server process
tasks_p = [] # create tasks processes
for k in xrange(T):
    tasks_p.append(Process(target=tasks[k].run()))
