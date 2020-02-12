include ../shared.mk

.PHONY: all lib spmm clean

.DEFAULT_GOAL := all

all: lib spmm

lib: dspmm_mgpu_baseline.o spmv_helper.o

spmm:  spmv_helper.o dspmm_mgpu_baseline.o ./test/dspmm_baseline_test.cu
	(cd test && $(NVCC) $(NVCC_FLAGS) ../src/spmv_helper.o ../src/dspmm_mgpu_baseline.o dspmm_baseline_test.cu -o test_spmm $(LINK_FLAG) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN))
	cp test/test_spmm ../

dspmm_mgpu_baseline.o: ./src/dspmm_mgpu_baseline.cu
	(cd src && $(NVCC) -c $(NVCC_FLAGS) dspmm_mgpu_baseline.cu $(LINK_FLAG))

spmv_helper.o: ./src/spmv_helper.cu 
	(cd src && $(NVCC) -c $(NVCC_FLAGS) spmv_helper.cu $(LINK_FLAG))

clean:
	(cd src && rm *.o)
	(cd test && rm spmm)

