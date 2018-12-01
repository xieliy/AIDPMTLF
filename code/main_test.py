from task_test import task_test
from server_test import server_test
from multiprocessing import Queue, Lock, Process

T = 2 # number of task
ITER = 10000
tasks = [] # create tasks
lock = Lock()
conn = Queue() # creating a Queue between tasks and server
server_ins = server_test(conn, lock, T) # create server
for k in xrange(T):
    tasks.append(task_test(conn, k, ITER))

server_p = Process(target=server_ins.run()) # create server process
server_p.daemon = True # server_p will stop after main process stop
server_p.start() # start from server

tasks_p = [] # create tasks processes
for k in xrange(T):
    tasks_p.append(Process(target=tasks[k].run()))
    tasks_p[-1].start()
