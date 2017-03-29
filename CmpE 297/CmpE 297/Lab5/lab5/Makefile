.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib 
CC	:= nvcc
DEP	:= reduction.h 
ifeq ($(def), )
DEF := 
else
DEF := -D$(def)
endif

NVCCFLAGS	:= -lineinfo -arch=sm_53 #--ptxas-options=-v -g 

all:	reduction

reduction:	cmpe297_reduction.o cmpe297_reduction_kernel.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o reduction cmpe297_reduction.o cmpe297_reduction_kernel.o $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@
	

clean:
	rm -f *.o reduction


