import sys
sys.path.insert(0, r'E:\course\caffe-master\Build\x64\Release\pycaffe')
import numpy as np
import caffe

caffe.set_mode_cpu()

net_file = 'custom_mnist_deploy.prototxt'
caffe_model = 'custom_iter_5000.caffemodel'
#mean_file = 'mnist.npy'

# load model
net = caffe.Net(net_file, caffe_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255./256.)
#transformer.set_channel_swap('data', (2, 1, 0))

im = caffe.io.load_image('image/3.png', 0)
print im.shape
net.blobs['data'].data[...] = transformer.preprocess('data', im)
out = net.forward()

labels_filename = 'labels.txt'
labels = np.loadtxt(labels_filename, str, delimiter='\t')

print net.blobs['prob'].data[0].shape
print net.blobs['prob'].data[0]

top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print '--------result--------'
print labels[top_k[0]]
#for i in np.arange(top_k.size):
    #print top_k[i], labels[top_k[i]]
#    print labels[top_k[i]]
