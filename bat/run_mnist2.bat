cd E:\course\caffe-master


"Build\x64\Release\caffe" ^
    train ^
    --solver=examples\mnist\lenet_solver.prototxt ^
    --snapshot=examples\mnist\lenet_iter_5000.solverstate ^
    2>&1 | "tools\mtee"  mnist.log

    rem --weights=examples\mnist\lenet_iter_5000.caffemodel

rem name of application program
rem tran
rem solver

pause