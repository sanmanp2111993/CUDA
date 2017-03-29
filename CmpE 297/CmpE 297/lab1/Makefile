.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib -lcudart
CC  := nvcc
OBJS    := deviceQuery.o
DEP :=

NVCCFLAGS   := -lineinfo -arch=sm_53 --ptxas-options=-v -g

all:	deviceQuery

deviceQuery:  $(OBJS) $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o deviceQuery $(OBJS) $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@
    

clean:
	rm -f *.o deviceQuery
