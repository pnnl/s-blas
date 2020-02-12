#!/bin/bash
PATH_TO_SMATRIX=./delaunay_n20.mtx

#./spmm $PATH_TO_SMATRIX 400 1 1
#./spmm $PATH_TO_SMATRIX 400 2 1
#./spmm $PATH_TO_SMATRIX 400 4 1
./spmm $PATH_TO_SMATRIX 400 8 1
