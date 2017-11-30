CC = nvcc

SRC = ./src
BIN = ./bin
MATLAB = /opt/MATLAB/R2014a

BINGCN = $(BIN)/gcn
MATLAB_BIN = $(MATLAB)/bin
MATLAB_INC = $(MATLAB)/extern/include
MATLAB_VER = $(MATLAB_BIN)/glnxa64

# options for release
CFLAGS = -lmat -lmx
CUDA_VERSION = -DCUDAVERSION0
XLINKER = -Xlinker
ARCH = -arch=sm_35

# source code
SOURCE = $(SRC)/hyper_version_mGPU_MBGD.cu

# object
OBJ = $(BIN)/main.o

all: $(BINGCN)

clean:
	rm $(BINGCN)

$(BINGCN): $(OBJ)
	$(CC) -o $(BINGCN) $(OBJ) $(XLINKER) -rpath=$(MATLAB_VER) -L$(MATLAB_VER) $(CFLAGS) -I$(MATLAB_INC) $(CUDA_VERSION)
	rm -rf $(BIN)/*.o

$(OBJ): $(SOURCE)
	$(CC) -o $(OBJ) -c $(ARCH) $(SOURCE) $(XLINKER) -rpath=$(MATLAB_VER) -L$(MATLAB_VER) $(CFLAGS) -I$(MATLAB_INC) $(CUDA_VERSION)
