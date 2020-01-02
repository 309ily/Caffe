#!/usr/bin/python 

import numpy as np
import sys
sys.path.insert(0, r'E:\course\caffe-master\Build\x64\Release\pycaffe')


import caffe 
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '../examples/mnist/mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( 'mean' , out )
