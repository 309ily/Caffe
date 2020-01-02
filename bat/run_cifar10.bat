cd E:\course\caffe-master


"Build\x64\Release\caffe" ^
    train ^
    --solver=examples\cifar10\cifar10_solver.prototxt
    
    
rem --snapshot=examples\mnist\lenet_iter_5000.solverstate
rem --weights=examples\mnist\lenet_iter_5000.caffemodel

rem name of application program
rem tran
rem solver

pause