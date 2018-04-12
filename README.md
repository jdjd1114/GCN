# GPU-based Cube CNN
# Add supprot for multiple GPUs in version 2.0

1. Install CUDA and nvcc compiler

2. Download GCN from https://github.com/jdjd1114/GCN
   <br>`bash`
   git clone https://github.com/jdjd1114/GCN.git
   <br>`bash`
   cd path_to_GCN/
   
3. Compile (Note that MATLAB path in Makefile may need to be modified.)
   <br>`bash`
   make
   
4. run the program
   <br>`bash` ./bin/gcn data/DATASET.mat device_id
   <br> DATASET.mat consists of three parts: 
   <br> * Dataset, a 3-dimensional hyperspectal image
   <br> * labels, a 2-dimensional label matrix
   <br> * Device ID

5. run GCN_version_2.0 ( Multiple GPUs verison )
   <br> modify SRC in Makefile (src/hyper_version_mGPU.cu)
   <br> recompile with Makefile
   <br>`bash` make
   <br>`bash` ./bin/gcn data/DATASET.mat device_num master_dev_id
