np=2
task=1
#do not using too large task size for small matrices.
#Total number of tasks = np * task

/home/lian599/opt/miniconda2/bin/mpirun -n $np -ppn $np ./test_sptrsv -n 1 -k $task -mtx ash85/ash85.mtx

