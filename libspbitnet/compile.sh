# if [ -z "$MySpMMPath" ]; then
#     # 如果为空，执行 source 命令
#     source ../../../test_env
# fi
nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --threads 0 \
    --shared spbitnet.cu \
    -lcuda -lcublas \
    -gencode arch=compute_86,code=sm_86 \
    -o libspbitnet.so

# nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared bitnet_kernels.cu -lcuda -gencode=arch=compute_80,code=compute_80 -o libbitnet.so


