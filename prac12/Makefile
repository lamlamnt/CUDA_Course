
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math -O3

all: kernel_overlap work_streaming work_streaming2

kernel_overlap:	kernel_overlap.cu Makefile
	nvcc kernel_overlap.cu -o kernel_overlap $(INC) $(NVCCFLAGS) $(LIB)

work_streaming:	work_streaming.cu Makefile
	nvcc --default-stream per-thread work_streaming.cu -o work_streaming $(INC) $(NVCCFLAGS) $(LIB)

work_streaming2:	work_streaming2.cu Makefile
	nvcc --default-stream per-thread work_streaming2.cu -o work_streaming2 $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f kernel_overlap work_streaming work_streaming2

