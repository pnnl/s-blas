#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cusparse.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "mmio.h"
#include <float.h>
#include <omp.h>
//#include "anonymouslib_cuda.h"
#include <cuda_profiler_api.h>
#include "spmv_kernel.h"
#include "spmm_kernel.h"
#include <limits>
using namespace std;
 
typedef struct{
    int r;
    int c;
    double v;
} rcv;


int cmp_func(const void *aa, const void *bb) {
    rcv * a = (rcv *) aa;
    rcv * b = (rcv *) bb;
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    return 0;
}


void sortbyrow(const int nnz, int * cooRowIndex, int * cooColIndex, double * cooVal) {
	rcv * rcv_arr = new rcv[nnz];
	for(int i = 0; i < nnz; ++i) {
		rcv_arr[i].r = cooRowIndex[i];
		rcv_arr[i].c = cooColIndex[i];
		rcv_arr[i].v = cooVal[i];
	}
	qsort(rcv_arr, nnz, sizeof(rcv), cmp_func);
	for(int i = 0; i < nnz; ++i) {
		cooRowIndex[i] = rcv_arr[i].r;
		cooColIndex[i] = rcv_arr[i].c;
		cooVal[i] = rcv_arr[i].v;
	}
	delete [] rcv_arr;
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed with error (%d) at line %d\n",                 \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed with error (%d) at line %d\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}



/*********For CUDA************/
void print_error(cusparseStatus_t status) {
	if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
		cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << endl;
	else if (status == CUSPARSE_STATUS_ALLOC_FAILED)
		cout << "CUSPARSE_STATUS_ALLOC_FAILED" << endl;
	else if (status == CUSPARSE_STATUS_INVALID_VALUE)
		cout << "CUSPARSE_STATUS_INVALID_VALUE" << endl;
	else if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
		cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << endl;
	else if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
		cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << endl;
	else if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
		cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << endl;
}

void print_dense_matrix(int m, int n, double * Val_A) {
	//cout << "======== A Dense Matrix ==============\n";
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			cout << Val_A[i + j * m] << '\t';
		}
		cout << endl;
	}
}

void print_sparse_matrix(int m, int n, int * csrRowPtr_A, int * csrColIndex_A, double * csrVal_A) {
	//cout << "======== print from print function ==============\n";
	//for (int i = 0; i < 19; ++i) {cout << csrColIndex_A[i] << " , " << csrVal_A[i] <<endl;}
	int row = 0;
	int col = 0;
	double * dA = new double[m*n];
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			dA[j + n*i] = 0.0;
		}
	}
	for (int i = 0; i < m; ++i) {
		for (int j = csrRowPtr_A[i]; j < csrRowPtr_A[i+1]; ++j) {
			dA[i*n + csrColIndex_A[j]] = csrVal_A[j];
			//cout << "r = " << i << ", c = " << csrColIndex_A[j] << " , val = " << csrVal_A[j] << endl;
		}
	}
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			cout << dA[j + i * n] << '\t';
		}
		cout << endl;
	}
	delete [] dA;
}


/*
int cusparse_single_gpu_mm101(	const int m,
				const int n,
				const int k,
				const double * alpha,
				const int nnz_A,
				int * csrRowPtr_A,
				int * csrColIndex_A,
				double * csrVal_A,
				const double * beta,
				double * B_dense,
				double * C_dense) {
// C = A * B
// C: m*n
// A: m*k
// B: k*n
	int * dev_csrRowPtr_A;
	int * dev_csrColIndex_A;
	double * dev_csrVal_A;

	double * dev_B_dense;
	double * dev_C_dense;

	CHECK_CUDA( cudaMalloc((void**) &dev_csrRowPtr_A, (m + 1) * sizeof(int)) )
	CHECK_CUDA( cudaMalloc((void**) &dev_csrColIndex_A, nnz_A * sizeof(int)) )
	CHECK_CUDA( cudaMalloc((void**) &dev_csrVal_A, nnz_A * sizeof(double)) )
	CHECK_CUDA( cudaMalloc((void**) &dev_B_dense, (k * n) * sizeof(double)) )
	CHECK_CUDA( cudaMalloc((void**) &dev_C_dense, (m * n) * sizeof(double)) )

	CHECK_CUDA( cudaMemcpy(dev_csrRowPtr_A, csrRowPtr_A,
			(m + 1) * sizeof(int),
			cudaMemcpyHostToDevice) )
	CHECK_CUDA( cudaMemcpy(dev_csrColIndex_A, 
			csrColIndex_A, nnz_A * sizeof(int),
			cudaMemcpyHostToDevice) )
	CHECK_CUDA( cudaMemcpy(dev_csrVal_A, csrVal_A,
			nnz_A * sizeof(double), 
			cudaMemcpyHostToDevice) )
	CHECK_CUDA( cudaMemcpy(dev_B_dense, B_dense, 
			(k * n) * sizeof(double),
			cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dev_C_dense, C_dense,
                        (m * n) * sizeof(double),
                        cudaMemcpyHostToDevice) )


	//--------------------------------------------------------------------------
	// CUSPARSE APIs
	cusparseHandle_t     handle = 0;
	cusparseSpMatDescr_t matA;
	cusparseDnMatDescr_t matB, matC;
	void*  dev_Buffer    = NULL;
	size_t bufferSize = 0;
	CHECK_CUSPARSE( cusparseCreate(&handle) )
	// Create sparse matrix A in CSR format
	CHECK_CUSPARSE( cusparseCreateCsr(
			&matA, m, n, nnz_A,
			dev_csrRowPtr_A, dev_csrColIndex_A, dev_csrVal_A,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )

	// Create dense mat B
	CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, k, dev_B_dense, 
						CUDA_R_64F, CUSPARSE_ORDER_COL) )
	// Create dense mat C
	CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, m, dev_C_dense,
                                                CUDA_R_64F, CUSPARSE_ORDER_COL) )
	// allocate an external buffer if needed
	CHECK_CUSPARSE( cusparseSpMM_bufferSize(handle, 
				CUSPARSE_OPERATION_NON_TRANSPOSE, 
				CUSPARSE_OPERATION_NON_TRANSPOSE,
                                alpha, matA, matB, beta, matC, CUDA_R_64F,
                                CUSPARSE_MM_ALG_DEFAULT, &bufferSize) )
	CHECK_CUDA( cudaMalloc(&dev_Buffer, bufferSize) )

	// execute SpMM
	CHECK_CUSPARSE( cusparseSpMM(handle, 
				CUSPARSE_OPERATION_NON_TRANSPOSE, 
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				alpha, matA, matB, beta, matC, CUDA_R_64F,
				CUSPARSE_MM_ALG_DEFAULT, &dev_Buffer) )

	// destroy mat descr
	CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
	CHECK_CUSPARSE( cusparseDestroyDnVec(matB) )
	CHECK_CUSPARSE( cusparseDestroyDnVec(matC) )
	CHECK_CUSPARSE( cusparseDestroy(handle) )

	// copy c to host
	CHECK_CUDA( cudaMemcpy(C_dense, dev_C_dense, (m * n) * sizeof(double),
				cudaMemcpyDeviceToHost) )

	CHECK_CUDA( cudaFree(dev_Buffer) )
	CHECK_CUDA( cudaFree(dev_C_dense) )
	CHECK_CUDA( cudaFree(dev_B_dense) )
	CHECK_CUDA( cudaFree(dev_csrVal_A) )
	CHECK_CUDA( cudaFree(dev_csrColIndex_A) )
	CHECK_CUDA( cudaFree(dev_csrRowPtr_A) )
	return 0;
}
*/

void print_dev_int(int * dec_d, int len) {
	int * host_d = new int[len];
	cudaMemcpy(host_d, dec_d, len * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < len; ++i) {cout << "[" << i << "] = " << host_d[i] << " ";}
	cout << endl;
	delete [] host_d;
}

void print_dev_double(double * dec_d, int len) {
        double * host_d = new double[len];
        cudaMemcpy(host_d, dec_d,
                        len * sizeof(double),
                        cudaMemcpyDeviceToHost);
        for (int i = 0; i < len; ++i) {cout << "[" << i << "] = " << host_d[i] << " ";}
        cout << endl;
        delete [] host_d;
}





int cusparse_single_gpu_csrmm(	const int m,
				const int n,
				const int k,
				const double * alpha,
				const int nnz_A,
				int * csrRowPtr_A,
				int * csrColIndex_A,
				double * csrVal_A,
				const double * beta,
				double * B_dense,
				double * C_dense) {
// C = A * B
// C: m*n
// A: m*k
// B: k*n
	cudaSetDevice(0);
	int * dev_csrRowPtr_A;
	int * dev_csrColIndex_A;
	double * dev_csrVal_A;

	double * dev_B_dense;
	double * dev_C_dense;
	
	cudaError_t cuda_status[5];
	cusparseStatus_t cusparse_status[2];

	cuda_status[0] = cudaMalloc((void**) &dev_csrRowPtr_A, (m + 1) * sizeof(int));
	cuda_status[1] = cudaMalloc((void**) &dev_csrColIndex_A, nnz_A * sizeof(int));
	cuda_status[2] = cudaMalloc((void**) &dev_csrVal_A, nnz_A * sizeof(double));
	cuda_status[3] = cudaMalloc((void**) &dev_B_dense, (k * n) * sizeof(double));
	cuda_status[4] = cudaMalloc((void**) &dev_C_dense, (m * n) * sizeof(double));
	
	for (int i = 0; i < 5; ++i) {
		if (cuda_status[i] != cudaSuccess) {
			cout << "cudaMalloc failed! \n"; 
			return 1;
		}
	}

	cuda_status[0] = cudaMemcpy(dev_csrRowPtr_A, csrRowPtr_A, 
				    (m + 1) * sizeof(int), 
				    cudaMemcpyHostToDevice);
	cuda_status[1] = cudaMemcpy(dev_csrColIndex_A, 
				    csrColIndex_A, nnz_A * sizeof(int),
				    cudaMemcpyHostToDevice);
	cuda_status[2] = cudaMemcpy(dev_csrVal_A, csrVal_A,
				    nnz_A * sizeof(double), 
				    cudaMemcpyHostToDevice);
	cuda_status[3] = cudaMemcpy(dev_B_dense, B_dense, 
				    (k * n) * sizeof(double),
				    cudaMemcpyHostToDevice);
        cuda_status[4] = cudaMemcpy(dev_C_dense, C_dense,
				    (m * n) * sizeof(double),
				    cudaMemcpyHostToDevice);
	
	for (int i = 0; i < 5; ++i) {
		if (cuda_status[i] != cudaSuccess) {
			cout << "cudaMemcpy cudaMemcpyHostToDevice failed! \n"; 
			return 1;
		}
	}
	
	//--------------------------------------------------------------------------
	// CUSPARSE APIs
	cusparseHandle_t     handle = 0;
	cusparseMatDescr_t     matA = 0;

	cusparse_status[0] = cusparseCreate(&handle);
	// Create sparse matrix A descr
	cusparse_status[1] = cusparseCreateMatDescr(&matA);
	
	if (cusparse_status[0] != CUSPARSE_STATUS_SUCCESS || 
	    cusparse_status[1] != CUSPARSE_STATUS_SUCCESS) {
		cout << "cusparseCreate failed! \n"; 
			return 1;
	}
		
	cusparseSetMatType(matA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(matA, CUSPARSE_INDEX_BASE_ZERO);
	

	// execute SpMM
	cusparse_status[0] = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					    m, n, k, nnz_A, alpha, matA,
					    dev_csrVal_A, dev_csrRowPtr_A, dev_csrColIndex_A,
					    dev_B_dense, k, beta,
					    dev_C_dense, m);
	if (cusparse_status[0] != CUSPARSE_STATUS_SUCCESS) {
		cout << "cusparseDcsrmm failed! \n"; 
			return 1;
	}

	// copy c to host
	cuda_status[0] = cudaMemcpy(C_dense, dev_C_dense, (m * n) * sizeof(double),
				    cudaMemcpyDeviceToHost);
	if (cuda_status[0] != cudaSuccess) {
		cout << "cudaMemcpy cudaMemcpyDeviceToHost failed! \n"; 
		return 1;
	}

        // destroy mat descr
        cusparse_status[0] = cusparseDestroyMatDescr(matA);
        cusparse_status[1] = cusparseDestroy(handle);
	if (cusparse_status[0] != CUSPARSE_STATUS_SUCCESS || 
	    cusparse_status[1] != CUSPARSE_STATUS_SUCCESS) {
		cout << "cusparseDestroy failed! \n"; 
			return 1;
	}

	cuda_status[0] = cudaFree(dev_C_dense);
	cuda_status[1] = cudaFree(dev_B_dense);
	cuda_status[2] = cudaFree(dev_csrVal_A);
	cuda_status[3] = cudaFree(dev_csrColIndex_A);
	cuda_status[4] = cudaFree(dev_csrRowPtr_A);
	
	for (int i = 0; i < 5; ++i) {
		if (cuda_status[i] != cudaSuccess) {
			cout << "cudaFree failed! \n"; 
			return 1;
		}
	}
		
	return 0;
}

int main(int argc, char * argv[]) {
	if (argc < 5) {
		std::cout << "Usage: ./spmm [input sparse matrix A file] [output row number] [number of GPU(s)] [number of test(s)]\n";
		return -1;
	}

	char * filename_A = argv[1];
	int n = atoi(argv[2]);
        int ngpu = atoi(argv[3]);
        int repeat_test = atoi(argv[4]);

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount < ngpu) {
		cout << "Error: Not enough number of GPUs. Only " << deviceCount << "available." << endl;
		return -1;
	}
	if (ngpu <= 0) {
		cout << "Error: Number of GPU(s) needs to be greater than 0." << endl;
		return -1;
	}
	cout << "Using " << ngpu << " GPU(s)." << endl; 


	MM_typecode matcode;
	int ret_code;
	FILE * f_A;
	int m, k;

	//Read in Matrix A
	int * cooRowIndex;
	int * cooColIndex;
	double * cooVal;
	int * csrRowPtr;
	int nnz;

	//cout << "Loading input matrix A from " << filename_A << "\n";

	if ((f_A = fopen(filename_A, "r")) == NULL) {
		cout << "Could not open matrix A file.\n";
		exit(1);
	}

	if (mm_read_banner(f_A, &matcode) != 0) {
		cout << "Could not process Matrix Market banner for matrix A.\n";
		exit(1);
	}

	if ((ret_code = mm_read_mtx_crd_size(f_A, &m, &k, &nnz)) != 0) {
		cout << "Could not read Matrix Market format for matrix A.\n";
		exit(1);
	}


	//double * A_dense = new double[m*k];
	//for (int i = 0; i <  m*k; ++i) { A_dense[i] = 0.0;}

        cout << "Matrix A -- #row: " << m << " #col: " << k << " nnz: " << nnz << endl;

        cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
        cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
        cudaMallocHost((void **)&cooVal, nnz * sizeof(double));

        cout << "Loading input matrix A from " << filename_A << "\n";
	int r_idx, c_idx;
	double value;
	for (int i = 0; i < nnz; ++i) {
		fscanf(f_A, "%d %d %lg\n", &r_idx, &c_idx, &value);
		//A_dense[(r_idx - 1) * m + c_idx - 1] = value;
		cooRowIndex[i] = r_idx - 1;  
	        cooColIndex[i] = c_idx - 1;
		cooVal[i] = value;
		//cout << "cooRowIndex[" << i << "] = " << cooRowIndex[i];
		//cout << ", cooColIndex[" << i << "] = " << cooColIndex[i];
		//cout << ", cooVal["  << i << "] = " <<  cooVal[i] << endl;
		
		if (cooRowIndex[i] < 0 || cooColIndex[i] < 0) { // report error
			cout << "i = " << i << " [" << cooRowIndex[i] << ", " << cooColIndex[i] << "] = " << cooVal[i] << endl;
		}
	}
	sortbyrow(nnz, cooRowIndex, cooColIndex, cooVal);


	// Convert COO to CSR
	//csrRowPtr = (int *) malloc((m+1) * sizeof(int));
	cudaMallocHost((void **)&csrRowPtr, (m+1) * sizeof(int));

	long long matrix_data_space = nnz * sizeof(double) + nnz * sizeof(int) + (m+1) * sizeof(int)
					+ (k * n + m * n) * sizeof(double);

	double matrix_size_in_gb = (double)matrix_data_space / 1e9;
	cout << "Matrix space size(total): " << matrix_size_in_gb << " GB." << endl;


	int * counter = new int[m];
	for (int i = 0; i < m; i++) { counter[i] = 0;}
	for (int i = 0; i < nnz; i++) {
		counter[cooRowIndex[i]]++;
	}
	//cout << "nnz: " << nnz << endl;
	//cout << "counter: ";
	int t = 0;
	for (int i = 0; i < m; i++) {
		//cout << counter[i] << ", ";
		t += counter[i];
	}

	//cout << "csrRowPtr: ";
	csrRowPtr[0] = 0;
	for (int i = 1; i <= m; i++) {
		csrRowPtr[i] = csrRowPtr[i - 1] + counter[i - 1];
		//cout << "csrRowPtr[" << i <<"] = "<<csrRowPtr[i] << endl;
	}
        delete [] counter;
	//print_sparse_matrix(m, k, csrRowPtr, cooColIndex, cooVal);

	
	cout << "Matrix B -- #row: " << k << " #col: " << n << " (dense)" << endl;
        cout << "Start generating data for Matrix B\n" << std::flush;

        double * B_dense;
        cudaMallocHost((void **)&B_dense, (k*n) * sizeof(double));
        for (int i = 0; i <  k*n; ++i) { B_dense[i] = (double) rand() / (RAND_MAX);}
        //print_dense_matrix(k, n, B_dense);

        double * C_dense;
        cudaMallocHost((void **)&C_dense, (m*n) * sizeof(double));
        for (int i = 0; i <  m*n; ++i) { C_dense[i] = (double) rand() / (RAND_MAX);}
	//cout << "initial C matrix: \n";
        //print_dense_matrix(m, n, C_dense);

        double * C_dense_mgpu;
        cudaMallocHost((void **)&C_dense_mgpu, (m*n) * sizeof(double));
        for (int i = 0; i < m*n; ++i) {
                C_dense_mgpu[i] = C_dense[i];
        }

        double alpha = -0.7;
        double beta = 0.8;


	cout << "Start computing SpMM on a single GPU (CuSPARSE).\n" << std::flush;

	double curr_time = get_time();
	cusparse_single_gpu_csrmm(m, n, k, &alpha, nnz, 
				  csrRowPtr, cooColIndex, cooVal, 
				  &beta, B_dense, C_dense);
	
	double single_gpu_time = get_time() - curr_time;
	cout << "CuSPARSE single gpu processing time(s): " << single_gpu_time << "\n";
	cout << "Matrix C -- #row: " << m << " #col: " << n << " (dense)" << endl;
	//print_dense_matrix(m, n, C_dense);
	
	curr_time = get_time();
    //cusparse_mgpu_csrmm(m, n, k, &alpha, nnz, 
	cusparse_mgpu_csrmm_omp(m, n, k, &alpha, nnz, 
			    csrRowPtr, cooColIndex, cooVal, 
			    &beta, B_dense, C_dense_mgpu, ngpu);
	
	double m_gpu_time = get_time() - curr_time;
	cout << "SPMM: " << ngpu << " GPUs processing time(s): " << m_gpu_time << "\n";


	bool checkflag = true;
	for (int i = 0; (i < m*n) && checkflag; ++i) {
		checkflag = (abs(C_dense_mgpu[i] - C_dense[i]) < 0.001);
	}

	cout << "mgpu check: " << (checkflag? "PASS":"FAILED") << endl;

	cudaFreeHost(cooRowIndex);
	cudaFreeHost(cooColIndex);
	cudaFreeHost(cooVal);
	cudaFreeHost(csrRowPtr);
	cudaFreeHost(C_dense);
	cudaFreeHost(B_dense);

	return 0;
}
