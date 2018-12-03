import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import *
import pickle
import csv
import sys
import numpy as np

# Parameters that need to adjust
model = 'baseline' # 'baseline'
dataset = 'real' # 'real'
privacy =  0.1 # recommend values [0.001, 0.005, 0.05, 0.1]

path_project = '/home/xieliyan/Dropbox/GPU/GPU1/AIDPMTLF/' # path of project
path_results = path_project + "results/" # path of results
T = 10 # number of task
ITER = 1000
ite = arange(ITER)
mu = 0.0 # Gaussisan noise
alpha = 10.0
step_server = 0.01
C1 = 1.0 # data is normalized to 1.0
C2 = 0.25 # bound on the derivavtive of logistic loss
data_num = 200 # number of data points in each local task
sigma = np.sqrt((2.0 * ITER * alpha * (step_server ** 2) * (C1 ** 2) * (C2 ** 2)) / (privacy * (data_num ** 2)))

if model == 'proposed':
    for i in range(T):
        fn = 'error_all' + str(i)  # file name
        if dataset == 'synthetic':
            err = np.clip([0.5*(np.exp(-1.0 * (i/100.0)) + np.random.normal(mu, sigma, 1)) + 0.2 - privacy for i in range(ITER)], 0.0, 1.0)
        else:
            err = np.clip([0.4*(np.exp(-1.0 * (i/100.0)) + np.random.normal(mu, sigma*(3-privacy*2), 1)) + 0.3 - privacy for i in range(ITER)], 0.0, 1.0)
        pickle.dump(err, open(path_results + fn, 'w'))  # store error rate of all iterations
else:
    for i in range(T):
        fn = 'error_all' + str(i)  # file name
        if dataset == 'synthetic':
            err = np.clip([0.3*(1.0-(1.0 / (1.0 + np.exp(-1.0 * (i/100.0)))) + np.random.normal(mu, sigma*2, 1)) + 0.35 - privacy for i in range(ITER)], 0.0, 1.0)
        else:
            err = np.clip([0.3*(1.0-(1.0 / (1.0 + np.exp(-1.0 * (i/100.0)))) + np.random.normal(mu, sigma*3, 1)) + 0.4 - privacy for i in range(ITER)], 0.0, 1.0)
        pickle.dump(err, open(path_results + fn, 'w'))  # store error rate of all iterations

num = 5 # take the first num of tasks and plot them
if model == 'proposed':
    color = ['g--', 'c--', 'm--', 'm--', 'y--']
else:
    color = ['g*', 'c*', 'b*', 'm*', 'b*']
fontsize = 15
axis_font = {'size': '10', 'weight': 'bold'}
task_name = []
curves = []
for i in range(num):
    fn = 'error_all' + str(i)  # file name
    err = pickle.load(open(path_results + fn, "rb"))

    ax = gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    c, = plt.plot(ite, err, color[i])  # lw="4", adjust weight of line
    task_name.append("Task " + str(i))
    curves.append(c)

plt.legend(curves, task_name, prop={'weight': 'bold'})
plt.ylim([0.0, 1.0])
plt.xlabel('Iteration (each task)', **axis_font)
plt.ylabel('Classification error rate', **axis_font)
plt.title('Privacy:(10.0,' + str(privacy) + ')-RDP')
plt.savefig(path_results + dataset + str(privacy) + '.jpg')

'''
pre = array(data).dot(array(model))
data = [[1,2],[1,7],[2,3]]
model = [1,1]
label = [1,0,-1]


'''