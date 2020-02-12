include ../../shared.mk

.PHONY: all clean

.DEFAULT_GOAL := all

all: ./src/main.cu
	$(NVCC) $(NVCC_FLAGS) ./src/main.cu -o test_sptrsv $(LINK_FLAG) -G -Xptxas -dlcm=cg -D VALUE_TYPE=$(VALUE_TYPE)
	cp test_sptrsv ../../test_sptrsv_v1

clean:
	(rm test_sptrsv)

