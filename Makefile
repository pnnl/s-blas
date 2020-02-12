include shared.mk

.PHONY: all clean

#all: test_spmv test_sptrsv test_sptrans test_spmm
all: test_spmv test_sptrsv test_spmm

test_spmv:
	(cd spmv && make)

test_sptrsv:
	(cd sptrsv/sptrsv_v1 && make)
	(cd sptrsv/sptrsv_v2 && make)
	(cd sptrsv/sptrsv_v3 && make)

test_sptrans:
	(cd sptrans/sptrans_v1 && make)
	(cd sptrans/sptrans_v2 && make)

test_spmm:
	(cd spmm && make)

clean:
	(rm test_spmv test_sptrsv* test_sptrans* test_spmm)



