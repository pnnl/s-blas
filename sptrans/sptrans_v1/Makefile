include ../../shared.mk

.PHONY: all clean

.DEFAULT_GOAL := all

all: ./src/main.cu
	$(NVCC) $(NVCC_FLAGS) ./src/main.cu -o sptrans $(LINK_FLAG) -D VALUE_TYPE=$(VALUE_TYPE) -g -G
	cp sptrans ../../test_sptrans_v1

clean:
	(rm sptrans)

