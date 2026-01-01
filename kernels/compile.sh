
nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --threads 0 \
    --shared TernaSpMM.cu \
    -lcuda -lcublas \
    -Icutlass/include   -Icutlass/tools/util/include \
    -w \
    -gencode arch=compute_86,code=sm_86 \
    -o libternaspmm.so



nvcc test_kernel.cu -L. -lternaspmm -lcublas -o test_kernel \
    -arch=sm_86 \
    -Xlinker -rpath -Xlinker '$ORIGIN'


