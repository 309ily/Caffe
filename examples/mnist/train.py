
import sys
sys.path.insert(0, r'E:\course\caffe-master\Build\x64\Release\pycaffe')

import caffe

caffe.set_mode_cpu()
solver=caffe.SGDSolver('python_lenet_solver.prototxt')
#solver.net.copy_from('lenet_iter_5000.caffemodel')
solver.solve()
