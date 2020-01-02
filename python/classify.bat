py -2  classify.py ^
    --model_def ../examples/mnist/mnist_deploy.prototxt ^
    --pretrained_model ../examples/mnist/lenet_iter_5000.caffemodel ^
    --images_dim 28,28 ^
    --mean_file mean.npy ^
    ../examples/mnist/image/0.png ^
    foo
    
pause