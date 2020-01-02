
import sys
sys.path.insert(0, r'E:\course\caffe-master\Build\x64\Release\pycaffe')
#sys.path.insert(0, r'C:\Python27\Lib')

import caffe
import numpy as np

class custom_layer(caffe.Layer):
    def setup(self, bottom, top):
        print 'custom_layer setup\n'
        
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
    
    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data
        
    def backward(self, top, propagate_down, bottom):
        if(propagate_down[0]):
            bottom[0].diff[...] = top[0].diff
            
            