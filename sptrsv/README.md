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

**Release Number:** 0.2

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
| CUDA | 9.0 or newer | [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) | USA | None |  
| OpenMP | 3.1 or newer |[https://www.openmp.org/](https://www.openmp.org/) | USA | None |  
| NVSHMEM | 0.3.0 or newer | [https://developer.nvidia.com/nvshmem](https://developer.nvidia.com/nvshmem) | USA | Early Access |
| MPICH | 1.4.1 or newer | [https://www.mpich.org/](https://www.mpich.org/) | USA | None |

Distribution Files
------------------
| File | Description |
| ---- | ------- |
| ./sptrsv\_v1 | The baseline implementation of SpTRSV using the synchronization-free algorithm. |
| ./sptrsv\_v2 | The optimazition of SpTRSV to balance the workload (nnz) on multiple GPUs. |
| ./sptrsv\_v3 | The nvshmem implementation of SpTRSV. Requiring nvshmem separately (not include)| 
    | ./src    | Contains source code for SpTRSV. | 
    | ./ash85  | test matrix files. |
    | Makefile | Used to compile the library and tests. |


Installation Instructions
-------------------------
For Unified Memory implementation (sptrsv\_v1 and sptrsv\_v2)

(1) In the ```Makefile```, edit the variable ```CUDA_INSTALL_PATH``` to match the CUDA installation directory and ```CUDA_SAMPLES_PATH``` to match the directory of CUDA samples that were installed with CUDA.

(2) Add ```<CUDA installation directory>/bin``` to the system's $PATH environment variable.

(3) Type ```make``` to compile the ./sptrsv.

For PGAS-based NVSHMEM implementation (sptrsv\_v3)

(1) Install nvshmem following the default instruction.

(2) Install MPI execution environment

(3) Setup ```NVSHMEM_HOME``` in the Makefile of sptrsv\_v3 to the install path of NVSHMEM, and ensure mpirun is available in path (or setup the path to mpirun in run.sh) 

(4) Type ```make``` to compile the ./sptrsv.


Test Cases
----------

* To verify the build was successful, a small test can be done: 
	* ```cd``` into where ```./sptrsv``` is. 
	* run ```./sptrsv -n 1 -rhs 1 -forward -mtx ./ash85/ash85.mtx ``` to test SpTRSV on a input matrix ash85.mtx on a single GPU. If the test was successful, it should output the test pass and run time without any error message.


User Guide
==========

#### Using the ./sptrsv

* ##### if use unified version  
	
 	```./sptrsv -n [#gpu] -k [#task] -rhs 1 -forward -mtx [A.mtx]```
    
    * ```[#gpu]```: The number of GPU(s) will be applied 
    * ```[#task]```: The number of tasks per GPUs
    * ```[A.mtx]```: It will load input matrix from the given path
    * ```-rhs 1```: remain parameter for futher extend the SpTRSV to SpTRSM (remove from shmem version)
    * ```-forward```: solve for L matrix. For U matrix using -backward (not test, remove from shmem version)

* ##### if use nvshmem version
 
  ```mpirun -n [#gpu] -ppn [#gpu] ./sptrsv -n 1 -k [#task] -mtx [A.mtx]```
    * ```[#gpu]```: The number of GPU(s) will be applied 
    * ```[#task]```: The number of tasks per GPUs
    * ```[A.mtx]```: It will load input matrix from the given path
    * The nvshmem can only be used for P2P connected GPUs

* ##### Output

The ./sptrsv will run tests based on imput matrix with options specified by users (in the common file). The exection time will be reported. The correctness of the output of SpTRSV are verified by comparing their results with the x_ref.
    

### Description of Main.cu

The main function is used to read the input matrix, transfer it to L matrix under csc compression.    

### Description of SpTRSV Kernels

All SpTRSV version perform the matrix solver operation:
    	L ∗ x = B
        
##### sptrsv_syncfree_serialref.h:
------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| cscColPtrTR |int *| Array of n+1 elements that contains the start of every row and the end of the last row plus one.|
| cscRowIdxTR |int *| Array of nnz row indices of the nonzero elements of matrix L.|
| cscValTR |double *| Array of nnz nonzero elements of matrix L as in CSC format.|
| m | int |Number of rows of the input matrix L. |
| n | int |Number of columns of the input matrix L. | 
| nnzTR | int | Number of nonzero elements in the input matrix L. | 
| substitution | int | forward of backward.|
| rhs | int| Number of col in X for SpTRSM (not implement) |
| x | double * | Vector x |
| b | double * | Vector B |
| x_ref | double * | the reference X generated by main.cu which is used to varification the result|
------------
| Output | type|  Description |
| ---- |----| ------- | 
| x | double * (default) |Vector x |

##### sptrsv_syncfree_cuda.h:
------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| cscColPtrTR |int *| Array of n+1 elements that contains the start of every row and the end of the last row plus one.|
| cscRowIdxTR |int *| Array of nnz row indices of the nonzero elements of matrix L.|
| cscValTR |double *| Array of nnz nonzero elements of matrix L as in CSC format.|
| m | int |Number of rows of the input matrix L. |
| n | int |Number of columns of the input matrix L. | 
| nnzTR | int | Number of nonzero elements in the input matrix L. | 
| substitution | int | forward of backward.|
| rhs | int | Number of col in X for SpTRSM (not implement) |
| opt | int | Number of warp defined in common |
| x | double * | Vector x |
| b | double * | Vector B |
| x_ref | double * | the reference X generated by main.cu which is used to varification the result |
| gflops | double * | performance |
| ngpu | int | Number of GPUs used |
------------
| Output | type|  Description |
| ---- |----| ------- | 
| x | double * (default) | Vector x |
| gflops | double * | performance |

