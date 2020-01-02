
import sys
sys.path.insert(0, r'E:\course\caffe-master\Build\x64\Release\pycaffe')

import numpy as np
import matplotlib.pyplot as plt

import caffe

caffe.set_mode_cpu()
solver = caffe.SGDSolver('E:/course/caffe-master/examples/mnist/python_lenet_solver.prototxt')

max_iter = 2000
test_interval = 20
loss = np.zeros(max_iter)
acc = np.zeros(max_iter / test_interval)

for it in range(max_iter):
    solver.step(1)

    loss[it] = solver.net.blobs['loss'].data

    if it % test_interval == 0:
        solver.test_nets[0].forward(start='conv1')
        accuracy=solver.test_nets[0].blobs['accuracy'].data
        print 'Iter:', it, 'accuracy: ',accuracy
        acc[it // test_interval] = accuracy

fig, ax_loss = plt.subplots()
ax_acc = ax_loss.twinx()
ax_loss.plot(np.arange(max_iter), loss, 'b')
ax_acc.plot(test_interval * np.arange(len(acc)), acc, 'r')
ax_loss.set_xlabel('iter')
ax_loss.set_ylabel('train_loss')
ax_acc.set_ylabel('test accuracy')
fig.savefig("loss.jpg")
plt.show()
