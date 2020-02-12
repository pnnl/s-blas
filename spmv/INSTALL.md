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

**Operating System & Version:** Ubuntu 16.04

**Required Disk Space:** 2.8MB (additional space is required for storing test input matrix files).

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
| ./src | Contains source code for different implementations of SpVM. | 
| ./include | Contains header files. | 
| ./test | Contains source code of mini test programs. |
| Makefile | Used to compile the library and tests. |



Installation Instructions
-------------------------

(1) In the ```Makefile```, edit the variable ```CUDA_INSTALL_PATH``` to match the CUDA installation directory and ```CUDA_SAMPLES_PATH``` to match the directory of CUDA samples that were installed with CUDA.

(2) Add ```<CUDA installation directory>/bin``` to the system's $PATH environment variable.

(3) Type ```make``` to compile the library and tests.

Test Cases
----------

* To verify the install was successful, a small test can be done: 
	* ```cd``` into the ```test``` folder. 
	* run ```./test_spmv g 200 1 1 1 ``` to test SpMV on a small randomly generated input matrix on a single GPU. If the test was successful, it should output the time comparison of all three implementations without any error message.

* To run a more comprehensive test, a test script is provided:
	* ```cd``` into the ```test``` folder.
	* Change the ```NGPU``` and ```KERNEL``` variables in ```batch_test.sh``` to be desired number of GPU(s) and kernel version.
	* run ```./batch_test.sh``` to start test.

User Guide
==========

#### Using the test binary

* ##### if use randomly generated input matrix 
	
 	```./spmv g [matri size n] [num. of GPU(s)] [num. of repeat test(s)] [kernel version (1, 2, or 3)]```
    
    * ```[matri size n]```: It will ramdomly generate (double floating point) a non-uniformly distributed input matrix with size n*n. The dimension can be arbitrary large as long as it can fit into the CPU RAM. However, since the baseline version is not optimized for large scale, it will fail to launch if the matrix is too large. SpMV version 1 uses static sheduleing rule which also bring limitation on matrix size but has less strict constrain than the baseline version. 
    * ```[num. of GPU(s)]```: The number of GPU(s) will be applied to the baseline version and version 1. SpMV version 2 will be optimum number of GPU(s) from 1 to the number GPU(s) specified. 
    * ```[num. of repeat test(s)]```: Number of repeat test to be run. 
    * ```[kernel version (1, 2, or 3)]```: 1: the regular sparse matrix-vector multiplication in Nvidia's cuSparse; 2: the optimized sparse matrix-vector multiplication in Nvidia's cuSparse; 3: the sparse matrix-vector multiplication implemented in CSR5. Kernel version will be applied to SpVM version 1 and 2 only. The baseline version will use the kernel version 1 only.

*  ##### if use input matrix from file

	```./spmv f [path to matrix file] [num. of GPU(s)] [num. of repeat test(s)] [kernel version (1, 2, or 3)] [data type ('f' or 'b')] ```. 
    * ```[path to matrix file]```: It will load input matrix from the given path. The matrix files can be obtained from: [The SuiteSparse Matrix Collection]( https://sparse.tamu.edu/). 
    For example: 
    	* [Rail4284](https://sparse.tamu.edu/MM/Mittelmann/rail4284.tar.gz)
    	* [Circuit5M](https://sparse.tamu.edu/MM/Freescale/circuit5M.tar.gz)
    	* [ASIC_680k](https://sparse.tamu.edu/MM/Sandia/ASIC_680k.tar.gz)
    * ```[data type ('f' or 'b')]```: Some matries are filled with floating point elements and some are filled with binary elements. Since we only implemented double floating point SpVM, we will convert and treat all of them as double floating point elements. This require users to sprcify the data type in original matrix. 
    * ```[num. of GPU(s)]```: The number of GPU(s) will be applied to the baseline version and version 1. SpMV version 2 will be optimum number of GPU(s) from 1 to the number GPU(s) specified. 
    * ```[num. of repeat test(s)]```: Number of repeat test to be run. 
    * ```[kernel version (1, 2, or 3)]```: 1: the regular sparse matrix-vector multiplication in Nvidia's cuSparse; 2: the optimized sparse matrix-vector multiplication in Nvidia's cuSparse; 3: the sparse matrix-vector multiplication implemented in CSR5. Kernel version will be applied to SpVM version 1 and 2 only. The baseline version will use the kernel version 1 only.

* ##### Output
The test binary will run tests on all three SpMV versions with options specified by users. The exection time of each run and the averge time will be reported. The correctness of the output of SpVM version 1 and 2 are verified by comparing their results with the output of the baseline version. If the baseline version failed to launch (e.g., run out of memory error), the comparion result will output as 'N/A', since no comparison can be done.
    
### Description of SpMV Kernels

All SpVM version perform the matrix-vector operation:
    	y = α ∗ A ∗ x + β ∗ y
        
##### spMV_mgpu_baseline:
------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| m | int |Number of rows of the input matrix A. |
| n | int|Number of columns of the input matrix A. | 
| nnz |long long| Number of nonzero elements in the input matrix A. | 
| alpha |double *|  Scalar used for multiplication.|
| csrVal |double *|  Array of nnz nonzero elements of matrix A as in CSR format.|
| csrRowPtr |long long *| Array of m+1 elements that contains the start of every row and the end of the last row plus one.|
| csrColIndex |int *|Array of nnz column indices of the nonzero elements of matrix A.|
| x | double * |Vector x |
| beta |double *|  Scalar used for multiplication.|
| y | double * |Vector y |
| ngpu | int |Number of GPU(s) to be used. |
------------
| Output | type|  Description |
| ---- |----| ------- | 
| y | double * |Vector y |

##### spMV_mgpu_v1:
------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| m | int |Number of rows of the input matrix A. |
| n | int|Number of columns of the input matrix A. | 
| nnz |long long| Number of nonzero elements in the input matrix A. | 
| alpha |double *|  Scalar used for multiplication.|
| csrVal |double *|  Array of nnz nonzero elements of matrix A as in CSR format.|
| csrRowPtr |long long *| Array of m+1 elements that contains the start of every row and the end of the last row plus one.|
| csrColIndex |int *|Array of nnz column indices of the nonzero elements of matrix A.|
| x | double * |Vector x |
| beta |double *|  Scalar used for multiplication.|
| y | double * |Vector y |
| ngpu | int |Number of GPU(s) to be used. |
| kernel | int |The computing kernel (1 - 3) to be used. 1: the regular sparse matrix-vector multiplication in Nvidia's cuSparse; 2: the optimized sparse matrix-vector multiplication in Nvidia's cuSparse; 3: the sparse matrix-vector multiplication implemented in CSR5. |
------------
| Output | type|  Description |
| ---- |----| ------- | 
| y | double * |Vector y |


##### spMV_mgpu_v2:

------------
| Input parameter | type|  Description |
| ---- |----| ------- | 
| m | int |Number of rows of the input matrix A. |
| n | int|Number of columns of the input matrix A. | 
| nnz |long long| Number of nonzero elements in the input matrix A. | 
| alpha |double *|  Scalar used for multiplication.|
| csrVal |double *|  Array of nnz nonzero elements of matrix A as in CSR format.|
| csrRowPtr |long long *| Array of m+1 elements that contains the start of every row and the end of the last row plus one.|
| csrColIndex |int *|Array of nnz column indices of the nonzero elements of matrix A.|
| x | double * |Vector x. |
| beta |double *|  Scalar used for multiplication.|
| y | double * |Vector y. |
| ngpu | int |Number of GPU(s) to be used. |
| kernel | int |The computing kernel (1 - 3) to be used. 1: the regular sparse matrix-vector multiplication in Nvidia's cuSparse; 2: the optimized sparse matrix-vector multiplication in Nvidia's cuSparse; 3: the sparse matrix-vector multiplication implemented in CSR5. |
| nb | int |Number of elements per task. |
| q | int |Number of Hyper-Q(s) on each GPU. |
------------
| Output | type|  Description |
| ---- |----| ------- | 
| y | double * |Vector y |


    
