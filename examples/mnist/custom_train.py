
import sys
sys.path.insert(0, r'E:\course\caffe-master\Build\x64\Release\pycaffe')
import custom_layer
import caffe
caffe.set_mode_cpu()
solver=caffe.SGDSolver('custom_lenet_solver.prototxt')
#solver.net.copy_from('lenet_iter_5000.caffemodel')
solver.solve()
