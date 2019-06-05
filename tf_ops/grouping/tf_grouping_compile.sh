#  -Wl,--no-as-needed -Wl,-rpath,/home/kshitij/installedPrograms/anaconda3/envs/sandbox/lib/python2.7/site-packages/tensorflow/libtensorflow_framework.so
TF_LOC=/home/kshitij/installedPrograms/anaconda3/envs/sandbox/lib/python2.7/site-packages/tensorflow

#/bin/bash
/usr/local/cuda-8.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
# g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I ${TF_LOC}/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I ${TF_LOC}/include -I /usr/local/cuda-8.0/include -I ${TF_LOC}/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L${TF_LOC} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
