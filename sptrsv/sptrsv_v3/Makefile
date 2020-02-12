include ../../shared.mk

.PHONY: all clean

.DEFAULT_GOAL := all

all: ./src/main.cu
	$(NVCC) $(NVCC_FLAGS) ./src/main.cu -dc -o sptrsv.o $(LINK_FLAG) -lnvshmem -I$(NVSHMEM_HOME)/include -L$(NVSHMEM_HOME)/lib  -Xptxas -dlcm=cg -D VALUE_TYPE=$(VALUE_TYPE) 
	$(NVCC) $(NVCC_FLAGS) sptrsv.o -o test_sptrsv $(LINK_FLAG) -lnvshmem -I$(NVSHMEM_HOME)/include -L$(NVSHMEM_HOME)/lib  -Xptxas -dlcm=cg -D VALUE_TYPE=$(VALUE_TYPE) 
	cp test_sptrsv ../../test_sptrsv_v3

clean:
	(rm test_sptrsv)

