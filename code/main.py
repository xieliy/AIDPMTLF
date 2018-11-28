# each task is a process, pass method to target, do not use join() to allow them run independently

import csv
from numpy import array

path_project = '/home/xieliyan/Dropbox/GPU/GPU1/AIDPMTLF/' # path of project
path_data = path_project + "data/" # path of data
path_results = path_project + "results/" # path of results
T = 2 # number of task
p_train = 0.3 # percentage of training set
Lambda = 10 ** -3 # regularization parameter
d = 28
data_all = [[] for i in xrange(T)] # data for all tasks
label_all = [[] for i in xrange(T)] # label for all tasks

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

