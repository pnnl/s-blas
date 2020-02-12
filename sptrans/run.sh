cd sptrans_v1 && make
./test_sptrans -n 2 -csr -mtx ../../sample_matrix/qh768.mtx 128 8 1
cd ../sptrans_v2 && make
./test_sptrans -n 2 -csr -mtx ../../sample_matrix/qh768.mtx 128 8 1
