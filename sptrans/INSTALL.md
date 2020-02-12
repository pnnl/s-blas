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

**Principle Investigator:** Shuaiwen Song (Shuaiwen.Song@pnnl.gov)

**General Area or Topic of Investigation:** Optimizing sparse Basic Linear Algebra Subprograms (BLAS) on modern multi-GPU systems.

**Release Number:** 1.0

Installation Guide
==================

The following sections detail the compilation, packaging, and installation of the software. Also included are test data and scripts to verify the installation was successful.

Environment Requirements
------------------------

**Programming Language:** CUDA C/C++

**Operating System & Version:** Ubuntu 18.04

**Required Disk Space:** 40MB (additional space is required for storing input matrix files).

**Required Memory:** Varies with different tests.

**Nodes / Cores Used:** One node with one or more Nvidia GPUs.

Dependencies
------------
| Name | Version | Download Location | Country of Origin | Special Instructions |
| ---- | ------- | ----------------- | ----------------- | -------------------- |
| GCC | 5.4.0 | [https://gcc.gnu.org/](https://gcc.gnu.org/) | USA | None |  
| CUDA[*] | 10.1.105 or newer | [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) | USA | None |  
| OpenMP | 3.1 or newer |[https://www.openmp.org/](https://www.openmp.org/) | USA | None |  

[*] We employ the latest sparse transposition function (cusparseCsr2cscEx2) in cuda SDK 10.1.105 for single GPU transposition. This function cannot be support by old Nvidia Driver and SDK.
We also test the old sparse transposition function (cusparseDcsr2csc) in cuda SDK 9.1 and we observe it cannot output the correct result. 

Distribution Files
------------------
| File | Description |
| ---- | ---------- | 
| ./Sptrans | The baseline implementation of SpTRANS using the cusparse algorithm. |
| ./Sptrans-sort | The implementation os SpTRANS using sort algorithm (in development). |
| run.sh | the linux script used for test with a set of benchmarks and #gpu |
    | ./src    | Contains source code for SpTrans. | 
    | ./ash85  | test matrix files. |
    | Makefile | Used to compile the library and tests. |

Installation Instructions
-------------------------

(1) In the ```Makefile```, edit the variable ```CUDA_INSTALL_PATH``` to match the CUDA installation directory and ```CUDA_SAMPLES_PATH``` to match the directory of CUDA samples that were installed with CUDA.

(2) Add ```<CUDA installation directory>/bin``` to the system's $PATH environment variable.

(3) Type ```make``` to compile the ./sptrans.

Test Cases
----------

* To verify the build was successful, a small test can be done: 
	* ```cd``` into where ```./sptrans``` is. 
	* run ```./sptrans -n 1 -csr -mtx ./ash85/ash85.mtx ``` to test SpTrans on a input matrix ash85.mtx on a single GPU. If the test was successful, it should output the test pass and run time without any error message.


User Guide
==========

#### Using the ./sptrans

* ##### if use input mtx matrix  
	
 	```./sptrans -n [#gpu] -csr -mtx [A.mtx]```
    
    * ```[#gpu]```: The number of GPU(s) will be applied 
    * ```[A.mtx]```: It will load input matrix from the given path
    * ```-csr```: the input matrix should be compressed in csr format
    
* ##### Output

The ./sptrans will run tests based on imput matrix with options specified by users (in the common file). The exection time will be reported. The correctness of the output of SpTRANS are verified by comparing their results with the output of cpu kernal.
    

### Description of Main.cu

The main function is used to read the input matrix, transfer it in CPU (transfer.h) to generate csc compression for ref.    

### Description of SpTrans Kernels

All SpTRANS version perform the matrix T operation:
    	X(csr) -> X(csc)
        
##### sptrans_cuda.h:
Single GPU version using cusparse algorithem

------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| m | int |Number of rows of the input matrix X. |
| n | int |Number of columns of the input matrix X. | 
| nnz | int | Number of nonzero elements in the input matrix L. | 
| csrRowPtr |int *| Array of m+1 elements that contains the start of every row and the end of the last row plus one.|
| csrColIdx |int *| Array of nnz Col indices of the nonzero elements of matrix X.|
| csrVal |double *| Array of nnz nonzero elements of matrix L as in CSR format.|
| cscRowIdx |int *| Array of nnz Row indices of the nonzero elements of matrix X.|
| cscColPtr |int *| Array of n+1 elements that contains the start of every Col and the end of the last Col plus one.|
| cscVal |double *| Array of nnz nonzero elements of matrix L as in CSC format.|
| cscRowIdx_ref |int *| Array of nnz Row indices of the nonzero elements of reference matrix X.|
| cscColPtr_ref |int *| Array of n+1 elements that contains the start of every Col and the end of the last Col plus one.|
| cscVal_ref |double *| Array of nnz nonzero elements of matrix L as in CSC format.|

------------
| Output | type|  Description |
| ---- |----| ------- | 
| cscRowIdx |int *| Array of nnz Row indices of the nonzero elements of matrix X.|
| cscColPtr |int *| Array of n+1 elements that contains the start of every Col and the end of the last Col plus one.|
| cscVal |double *| Array of nnz nonzero elements of matrix L as in CSC format.|

##### sptrans_kernal.h:
Multiple GPUs version 

------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| m | int |Number of rows of the input matrix X. |
| n | int |Number of columns of the input matrix X. | 
| nnz | int | Number of nonzero elements in the input matrix L. | 
| ngpu | int | Number of GPUs used |
| csrRowPtr |int *| Array of m+1 elements that contains the start of every row and the end of the last row plus one.|
| csrColIdx |int *| Array of nnz Col indices of the nonzero elements of matrix X.|
| csrVal |double *| Array of nnz nonzero elements of matrix L as in CSR format.|
| cscRowIdx |int *| Array of nnz Row indices of the nonzero elements of matrix X.|
| cscColPtr |int *| Array of n+1 elements that contains the start of every Col and the end of the last Col plus one.|
| cscVal |double *| Array of nnz nonzero elements of matrix L as in CSC format.|
| cscRowIdx_ref |int *| Array of nnz Row indices of the nonzero elements of reference matrix X.|
| cscColPtr_ref |int *| Array of n+1 elements that contains the start of every Col and the end of the last Col plus one.|
| cscVal_ref |double *| Array of nnz nonzero elements of matrix L as in CSC format.|

------------
| Output | type|  Description |
| ---- |----| ------- | 
| cscRowIdx |int *| Array of nnz Row indices of the nonzero elements of matrix X.|
| cscColPtr |int *| Array of n+1 elements that contains the start of every Col and the end of the last Col plus one.|
| cscVal |double *| Array of nnz nonzero elements of matrix L as in CSC format.|

