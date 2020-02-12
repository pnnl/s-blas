import os
import sys
import string
import commands

#number of gpus to be tested
n_gpus = 2 
#matrix data path
#mtxpath = "/raid/data/SuiteSparse/test_matrices/" 
mtxpath = "./sample_matrix/" 
#matrix file
mtxfile = open("matrices.txt","r")
#matrix to be tested
mtxlist = []
#save result file
result_file = open("results.csv","w")
#mpirun path
mpipath = "/home/lian599/opt/miniconda2/bin/"

for line in mtxfile.readlines():
    l = line.strip()
    mtxlist.append(l)

print "The following matrices will be tested:"
print mtxlist


#================================== SPMV ==================================
def parse_spmv(result):
    for line in result.strip().split("\n"):
        l = line.strip()
        if l.startswith("m:"):
            words = l.strip("\n").split(" ")
            m = int(words[1])
            n = int(words[3])
            nnz = int(words[5])
        if l.startswith("Average"):
            words = l.strip("\n").split(" ") 
            words = [i for i in words if i]
            v1_time = float(words[1])
            v2_time = float(words[2])
            v3_time = float(words[3])
    return m, n, nnz, v1_time, v2_time, v3_time

def test_spmv():
    result_file.write("kernel, matrix, n_gpu, m, n, nnz, version, time\n")
    for mtx in mtxlist:
        for gpu in range(1, n_gpus+1):
            cmd = "./test_spmv f " + mtxpath + mtx + " " + str(gpu) + " 1 " \
                + "1 f"
            print cmd
            result = commands.getoutput(cmd)
            m, n, nnz, v1_time, v2_time, v3_time = parse_spmv(result)
            res = str("spmv, ") + mtx + ", " + str(gpu) + ", " +  \
                    str(m) + ", " + str(n) + ", " + str(nnz) + \
                    ", " + "V1" + ", " + str(v1_time) + "\n"
            result_file.write(res)
            res = str("spmv, ") + mtx + ", " + str(gpu) + ", " +  \
                    str(m) + ", " + str(n) + ", " + str(nnz) + \
                    ", " + "V2" + ", " + str(v2_time) + "\n"
            result_file.write(res)
            res = str("spmv, ") + mtx + ", " + str(gpu) + ", " +  \
                    str(m) + ", " + str(n) + ", " + str(nnz) + \
                    ", " + "V3" + ", " + str(v3_time) + "\n"
            result_file.write(res)
    result_file.write("\n")

#================================== SPTRSV ==================================
def parse_sptrsv(result):
    for line in result.strip().split("\n"):
        l = line.strip()
        if l.startswith("input matrix A:"):
            words = l.strip("\n").split(" ")
            m = int(words[4][:-1])
            n = int(words[5])
            nnz = int(words[9])
        if l.startswith("cuda syncfree SpTRSV solve used"):
            words = l.strip("\n").split(" ") 
            v1_time = float(words[5])
    return m, n, nnz, v1_time

def test_sptrsv():
    result_file.write("kernel, matrix, n_gpu, m, n, nnz, version, time\n")
    for mtx in mtxlist:
        for gpu in range(1, n_gpus+1):
            for lu in ['-forward']:
                for v in ['v1','v2']:
                    cmd = "./test_sptrsv_" + v + " -n " + str(gpu) + " -rhs 1 " + lu\
                            + " -mtx " + mtxpath + mtx
                    print cmd
                    result = commands.getoutput(cmd)
                    m, n, nnz, v1_time = parse_sptrsv(result)
                    res = str("sptrsv, ") + mtx + ", " + str(gpu) + ", " +  \
                            str(m) + ", " + str(n) + ", " + str(nnz) + \
                            ", " + v.upper() + ", " + str(v1_time) + "\n"
                    result_file.write(res)

            cmd = mpipath + "mpirun " + " -n " + str(gpu) + " -ppn " +\
                    str(gpu) + " ./test_sptrsv_v3" +\
                    " -n 1 -k 1 " + " -mtx " + mtxpath + mtx
            print cmd
            result = commands.getoutput(cmd)
            m, n, nnz, v1_time = parse_sptrsv(result)
            res = str("sptrsv, ") + mtx + ", " + str(gpu) + ", " +  \
                    str(m) + ", " + str(n) + ", " + str(nnz) + \
                    ", " + v.upper() + ", " + str(v1_time) + "\n"
            result_file.write(res)

    result_file.write("\n")
     


#================================== SPTRANS ==================================
def parse_sptrans(result):
    for line in result.strip().split("\n"):
        l = line.strip()
        if l.startswith("input matrix A:"):
            words = l.strip("\n").split(" ")
            m = int(words[4][:-1])
            n = int(words[5])
            nnz = int(words[9])
        if l.startswith("SpTrans computation time:"):
            words = l.strip("\n").split(" ") 
            #print words
            v1_time = float(words[3])
    return m, n, nnz, v1_time

def test_sptrans():
    result_file.write("kernel, matrix, n_gpu, m, n, nnz, version, time\n")
    for mtx in mtxlist:
        for gpu in range(1, n_gpus+1):
            for v in ['v1','v2']:
                cmd = "./test_sptrans_" + v + " -n " + str(gpu) +\
                        " -csr " + " -mtx " + mtxpath + mtx
                print cmd
                result = commands.getoutput(cmd)
                m, n, nnz, v1_time = parse_sptrans(result)
                res = str("sptrans, ") + mtx + ", " + str(gpu) + ", " +  \
                        str(m) + ", " + str(n) + ", " + str(nnz) + \
                        ", " + v.upper() + ", " + str(v1_time) + "\n"
                result_file.write(res)
    result_file.write("\n")


#================================== SPMM ==================================
def parse_spmm(result):
    for line in result.strip().split("\n"):
        l = line.strip()
        if l.startswith("Matrix A --"):
            words = l.strip("\n").split(" ")
            m = int(words[4])
            n = int(words[6])
            nnz = int(words[8])
        if l.startswith("Matrix B --"):
            words = l.strip("\n").split(" ")
            k = int(words[6])
        if l.startswith("SPMM:"):
            words = l.strip("\n").split(" ") 
            v1_time = float(words[5])
    return m, n, k, nnz, v1_time

def test_spmm():
    result_file.write("kernel, matrix, n_gpu, m, n, k, nnz, version, time\n")
    for mtx in mtxlist:
        for gpu in range(1, n_gpus+1):
            cmd = "./test_spmm " + mtxpath + mtx + " 128 " + str(gpu) + " 1 "
            print cmd
            result = commands.getoutput(cmd)

            m, n, k, nnz, v1_time = parse_spmm(result)
            res = str("spmm, ") + mtx + ", " + str(gpu) + ", " +  \
                    str(m) + ", " + str(n) + ", " + str(k) + ", " + str(nnz) + \
                    ", " + "V1" + ", " + str(v1_time) + "\n"
            result_file.write(res)
    result_file.write("\n")
 


test_spmv()
#test_sptrsv()
#test_sptrans()
test_spmm()




result_file.close()
