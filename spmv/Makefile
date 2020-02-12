include ../shared.mk

.PHONY: all lib test clean

.DEFAULT_GOAL := all

all: lib test

#lib: dspmv_mgpu_v2.o dspmv_mgpu_v1.o dspmv_mgpu_baseline.o csr5_kernel.o spmv_helper.o
lib: dspmv_mgpu_v2.o dspmv_mgpu_v1.o dspmv_mgpu_baseline.o spmv_helper.o

test: lib dspmv_test.o
	(cd test && $(NVCC) $(NVCC_FLAGS) ../src/spmv_helper.o dspmv_test.o ../src/dspmv_mgpu_baseline.o ../src/dspmv_mgpu_v1.o ../src/dspmv_mgpu_v2.o -o test_spmv $(LINK_FLAG) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN))
	cp test/test_spmv ../

dspmv_mgpu_v2.o: ./src/dspmv_mgpu_v2.cu 
	(cd src && $(NVCC) -c $(NVCC_FLAGS) dspmv_mgpu_v2.cu $(LINK_FLAG))

dspmv_mgpu_v1.o: ./src/dspmv_mgpu_v1.cu 
	(cd src && $(NVCC) -c $(NVCC_FLAGS) dspmv_mgpu_v1.cu $(LINK_FLAG))

dspmv_mgpu_baseline.o: ./src/dspmv_mgpu_baseline.cu 
	(cd src && $(NVCC) -c $(NVCC_FLAGS) dspmv_mgpu_baseline.cu $(LINK_FLAG))

dspmv_test.o: ./test/dspmv_test.cu 
	(cd test && $(NVCC) -c $(NVCC_FLAGS) dspmv_test.cu $(LINK_FLAG))

spmv_helper.o: ./src/spmv_helper.cu 
	(cd src && $(NVCC) -c $(NVCC_FLAGS) spmv_helper.cu $(LINK_FLAG))

clean:
	(cd src && rm *.o)
	(cd test && rm *.o)
	(cd test && rm test_spmv)
