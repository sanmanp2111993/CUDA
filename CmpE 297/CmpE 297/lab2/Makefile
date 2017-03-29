.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib 
CC	:= nvcc
OBJS	:= cmpe297_vecAdd.o
DEP	:=  

NVCCFLAGS	:= -lineinfo -arch=sm_53 -g

all:	vectorAdd

vectorAdd:	$(OBJS) $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o vectorAdd $(OBJS) $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@
	

clean:
	rm -f *.o vectorAdd


