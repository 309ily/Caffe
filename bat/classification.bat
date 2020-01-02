cd E:\course\caffe-master

"Build\x64\Release\classification" ^
    examples\mnist\mnist_deploy.prototxt ^
    examples\mnist\lenet_iter_5000.caffemodel ^
    examples\mnist\mean.binaryproto ^
    examples\mnist\labels.txt ^
    examples\mnist\image\4.png

pause