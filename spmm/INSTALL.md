Table of Contents
=================

*   [Project Overview](#project-overview)
    *   [Detailed Summary](#detailed-summary)
*   [Installation Guide](#installation-guide)
    *   [Environment Requirements](#environment-requirements)
    *   [Dependencies](#dependencies)
    *   [Distribution Files](#distrubution-files)
    *   [Installation Instructions](#installation-instructions)
    *   [Test Cases](#test-cases)
*   [User Guide](#user-guide)

Project Overview
================

**Project Name:** Sparse-BLAS

**Principle Investigator:** Ang Li (ang.li@pnnl.gov)

**Developers:** Chenhao Xie (chenhao.xie@pnnl.gov), Jieyang Chen (chenj3@ornl.gov), Jiajia Li (jiajia.li@pnnl.gov), Shuaiwen Song (shuaiwen.song@pnnl.gov), Linghao Song (linghao.song@pnnl.gov) Jesun Firoz (jesun.firoz@pnnl.gov)

**General Area or Topic of Investigation:** Implementing and optimizing sparse Basic Linear Algebra Subprograms (BLAS) on modern multi-GPU systems.

**Release Number:** 0.1

Installation Guide
==================

The following sections detail the compilation, packaging, and installation of the software. Also included are test data and scripts to verify the installation was successful.

Environment Requirements
------------------------

**Programming Language:** CUDA C/C++

**Operating System & Version:** Ubuntu 18.04

**Required Disk Space:** 42MB (additional space is required for storing input matrix files).

**Required Memory:** Varies with different tests.

**Nodes / Cores Used:** One node with one or more Nvidia GPUs.

Dependencies
------------
| Name | Version | Download Location | Country of Origin | Special Instructions |
| ---- | ------- | ----------------- | ----------------- | -------------------- |
| GCC | 5.4.0 | [https://gcc.gnu.org/](https://gcc.gnu.org/) | USA | None |  
| CUDA | 9.0 or newer | [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) | USA | None |  
| OpenMP | 3.1 or newer |[https://www.openmp.org/](https://www.openmp.org/) | USA | None |  

Distribution Files
------------------
| File | Description |
| ---- | ------- |
| ./SpMM | Multi-GPU implementation of sparse matrix-dense matrix multiplication. |
    | ./src    | Contains source code for SpMM. | 
    | ./test  | Driver program to test single-gpu and multi-gpu SpMM. |
    | ./include | Header functions |
    | Makefile | Makefile to compile the test driver. |


Installation Instructions
-------------------------

(1) In the ```Makefile```, edit the variable ```CUDA_INSTALL_PATH``` to match the CUDA installation directory and ```CUDA_SAMPLES_PATH``` to match the directory of CUDA samples that were installed with CUDA.

(2) Add ```<CUDA installation directory>/bin``` to the system's $PATH environment variable.

(3) Type ```make``` in the root directory. This will compile the driver in the ```test``` repo and will create a binary ```spmm```

Testing
----------

* Download any matrix-market formatted (.mtx or .mmio) sparse matrix data from [https://sparse.tamu.edu/](https://sparse.tamu.edu/). 

* To verify whether the build was successful, run the following command: 
	* ```cd``` to  ```./test``` folder. 
	* run ```./spmm ./delaunay_n20.mtx 4 2 1 ``` to test SpMM on a sample input matrix delaunay_n20 on 4 GPUs. If the test was successful, it should output the test pass and run time without any error message.


User Guide
==========

#### Using the SpMM

* ##### if use input mtx matrix  
	
 	```./spmm [input sparse matrix A file] [output column number] [number of GPU(s)] [number of test(s)]```
    
    * ```[number of GPU(s)]```: The number of GPU(s) to run on 
    * ```[input sparse matrix A file]```: the input sparse matrix to be loaded from the given path
    * ```[output column number]```: Number of columns for the dense matrix
    * ```[number of test(s)]```: No of times the test will be executed with the same input.
    
* ##### Output

The ./spmm will first report the execution time of the SpMM operation with the given input using CuSPARSE library and single GPU. Next it will report the execution time with the specified number of GPUs. In addition, a breakdown of total multi-gpu execution time is also reported in terms of compute time, communication time, and post-processing time.
    

### Description of dspmm_baseline_test.cu

The driver program reads the sparse matrix provided in mmio format, stores it in a compressed sparse row (CSR) data structure on the  host, executes CuSPARSE single-gpu and our multi-gpu implementation, and report execution time in both cases.    

### Description of SpMM Kernels

SpMM multiplies two matrices, where matrix A is sparse and matrix B is dense:
    	A * B = C
        
##### dspmm_mgpu_baseline.cu:
This file contains the main function ```cusparse_mgpu_csrmm``` to perform the muti-gpu multiplication. In the multi-gpu version, matrix A (the sparse matrix) is copied from the host to each GPU, and the columns of matrix B is partitioned evenly to each GPU. Each gpu computes its part of the multiplication, and at the end of the execution the results are combined in matrix C. The function allocates device memory for the CSR representation of the matrix A , as well as device memory for the part of matrix B and C on each device. The memcpys are done in separate streams for each device for faster allocation. Once memory is allocated, CuSPARSE function ```cusparseDcsrmm``` is called on each device to perform multiplication on each device. Once the multiplication kernels finish execution, the result is copied back to the host.

------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| m | int| Number of rows in matrix A|
| k | int| Number of columns in matrix A and number of rows in matrix B|
| n | int| Number of columns in matrix B|


| alpha | double *|scalar used for multiplication|
| beta | double *|scalar used for multiplication|
| host_csrRowPtr_A | int *|pointers to the beginning of the rows in the CSR representation of matrix A on the host. |
| host_csrColIndex_A | int *| Column indices in the CSR representation of matrix A on the host|
| host_csrVal_A | double *| Non-zero values stored in the sparse matrix A on the host|
| host_B_dense | double *| Dense matrix B on the host|
| ngpu |int| Number of GPUs to be used|

------------
| Output | type|  Description |
| ---- |----| ------- | 
| host_C_dense | double *| Resultant dense matrix C|
