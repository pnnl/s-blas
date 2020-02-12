#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
#include <omp.h>
//#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"
#include "spmm_kernel.h"
using namespace std;


/** Error Checking **/

#define CUDA_SAFE_CALL( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
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

int cusparse_mgpu_csrmm_omp(const int m,
        const int n,
        const int k,
        const double * alpha,
        const int nnz_A,
        int * host_csrRowPtr_A,
        int * host_csrColIndex_A,
        double * host_csrVal_A,
        const double * beta,
        double * host_B_dense,
        double * host_C_dense,
        const int ngpu)
{
    /***************************************************/
    // C = A * B
    // A: m*k, sparse

    // B: k*n, dense, column major
    // C: m*n, dense, column major


//Start OpenMP
#pragma omp parallel num_threads (ngpu)
    {
        int dev = omp_get_thread_num();
        CUDA_SAFE_CALL( cudaSetDevice(dev) );
        cusparseHandle_t handle;
        cusparseMatDescr_t  mat_A;
        double curr_time = 0.0;
        double time_parse = 0.0;
        double time_alloc = 0.0;
        double time_comm = 0.0;
        double time_comp = 0.0;
        double time_post = 0.0;

        int *dev_csrRowPtr_A = NULL;
        int *dev_csrColIndex_A = NULL;
        double *dev_csrVal_A = NULL;
        double *dev_B_dense = NULL;
        double *dev_C_dense = NULL;
        cusparseStatus_t cusparse_status;
        int dev_n = 0;
        int B_offset = 0;
        int C_offset = 0;

        /*
        //The method here is to duplicate sparse matrix A while sharing 
        //dense matrix B among ngpus. It is applicable to the case when
        //k in (m*n)x(n*k) is large.
        double matrix_size_in_gb = 1e-9*( (nnz_A * sizeof(double) 
                + nnz_A * sizeof(int) + (m + 1) * sizeof(int)) * ngpu
                + k * n * sizeof(double) / ngpu 
                + m * n * sizeof(double) / ngpu);

        if ( matrix_size_in_gb > get_gpu_availble_mem(ngpu)) 
        {
            cout << "No available device memory for " 
                << matrix_size_in_gb << " GB \n";
            exit(-1);
        }
        */

        curr_time = get_time();

        dev_n = floor((dev + 1) * n / ngpu) - floor(dev * n / ngpu);

        B_offset = floor(dev * n / ngpu) * k;
        C_offset = floor(dev * n / ngpu) * m;


        ////#pragma omp barrier
        //if (dev == 0)
        //{
        //time_parse = get_time() - curr_time;
        //cout << "parse time for " << ngpu << 
        //" GPUs: " << time_parse << endl;
        //}
        //curr_time = get_time();


		cusparse_status = cusparseCreate(&handle); 

		if (cusparse_status != CUSPARSE_STATUS_SUCCESS) 
        { 
			printf("CUSPARSE Library initialization failed\n");
			exit(-1); 
		} 

		cusparse_status = cusparseCreateMatDescr(&mat_A);
		if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
			printf("Matrix descriptor initialization failed\n");
            exit(-1);
		} 	

		cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO); 

        CUDA_SAFE_CALL( cudaMalloc((void**)&dev_csrRowPtr_A, 
                    (m + 1) * sizeof(int)) );
        CUDA_SAFE_CALL( cudaMalloc((void**)&dev_csrColIndex_A, 
                    nnz_A * sizeof(int)) );
        CUDA_SAFE_CALL( cudaMalloc((void**)&dev_csrVal_A, 
                    nnz_A * sizeof(double)) );
        CUDA_SAFE_CALL( cudaMalloc((void**)&dev_B_dense, 
                    k * dev_n * sizeof(double)) );
        CUDA_SAFE_CALL( cudaMalloc((void**)&dev_C_dense, 
                    m * dev_n * sizeof(double)) );

    #pragma omp barrier
        if (dev == 0)
        {
            time_alloc = get_time() - curr_time;
            cout << "alloc time for " << ngpu << 
                " GPUs: " << time_alloc << endl;
        }
        curr_time = get_time();


        CUDA_SAFE_CALL( cudaMemcpy(dev_csrRowPtr_A, host_csrRowPtr_A,
                (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_csrColIndex_A, host_csrColIndex_A,
                nnz_A * sizeof(int), cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_csrVal_A, host_csrVal_A,
                nnz_A * sizeof(double), cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_B_dense, 
                    host_B_dense + (size_t)B_offset,
                    k * dev_n * sizeof(double), 
                    cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_C_dense, 
                    host_C_dense + (size_t)C_offset,
                    m * dev_n * sizeof(double), 
                    cudaMemcpyHostToDevice) );

    #pragma omp barrier
        if (dev == 0)
        {
            time_comm = get_time() - curr_time;
            cout << "comm time for " << ngpu << " GPUs: " << time_comm << endl;
        }
    #pragma omp barrier

        curr_time = get_time();
        cusparse_status = cusparseDcsrmm(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                m,
                dev_n,
                k,
                nnz_A,
                alpha,
                mat_A,
                dev_csrVal_A,
                dev_csrRowPtr_A,
                dev_csrColIndex_A,
                dev_B_dense,
                k,
                beta,
                dev_C_dense,
                m);		 	 	

        cudaDeviceSynchronize(); 
        CUDA_CHECK_ERROR();
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            cout << "cusparseDcsrmm for device " << dev << " failed! \n";
            exit(-1);
        }
    #pragma omp barrier
        if (dev == 0)
        {
            time_comp = get_time() - curr_time;
            cout << "comp time for " << ngpu << " GPUs: " << time_comp << endl;
        }

        curr_time = get_time();

        CUDA_SAFE_CALL( cudaMemcpy(host_C_dense + (size_t)C_offset,
                dev_C_dense,
                (size_t)(m * dev_n * sizeof(double)),
                cudaMemcpyHostToDevice) );

        cusparseDestroyMatDescr(mat_A);
        cusparseDestroy(handle);

        CUDA_SAFE_CALL( cudaFree(dev_csrVal_A) );
        CUDA_SAFE_CALL( cudaFree(dev_csrRowPtr_A) );
        CUDA_SAFE_CALL( cudaFree(dev_csrColIndex_A) );
        CUDA_SAFE_CALL( cudaFree(dev_B_dense) );
        CUDA_SAFE_CALL( cudaFree(dev_C_dense) );

    #pragma omp barrier
        if (dev == 0)
        {
            time_post = get_time() - curr_time;
            cout << "post time for " << ngpu << " GPUs: " << time_post << endl;
        }
    }

	return 0;
}

int cusparse_mgpu_csrmm(const int m,
        const int n,
        const int k,
        const double * alpha,
        const int nnz_A,
        int * host_csrRowPtr_A,
        int * host_csrColIndex_A,
        double * host_csrVal_A,
        const double * beta,
        double * host_B_dense,
        double * host_C_dense,
        const int ngpu){
    /***************************************************/
    // C = A * B
    // A: m*k, sparse

    // B: k*n, dense, column major
    // C: m*n, dense, column major

    double curr_time = 0.0;
    double time_parse = 0.0;
    double time_comm = 0.0;
    double time_comp = 0.0;
    double time_post = 0.0;

    cudaStream_t * stream = new cudaStream_t [ngpu];

    cusparseHandle_t * handle   = new cusparseHandle_t[ngpu];
    cusparseMatDescr_t * mat_A = new cusparseMatDescr_t[ngpu];

    int ** dev_csrRowPtr_A   = new int    * [ngpu];
    int ** dev_csrColIndex_A = new int    * [ngpu];
    double ** dev_csrVal_A   = new double * [ngpu];

    double ** dev_B_dense   = new double * [ngpu];
    double ** dev_C_dense   = new double * [ngpu];

    cudaError_t ** cuda_status = new cudaError_t*[ngpu];
    cusparseStatus_t ** cusparse_status = new cusparseStatus_t*[ngpu];

    int * dev_n = new int[ngpu];
    int * B_offset = new int[ngpu];
    int * C_offset = new int[ngpu];

    curr_time = get_time();

    double matrix_size_in_gb = 1.2 / 1e9;
    matrix_size_in_gb *= (nnz_A * sizeof(double) + nnz_A * sizeof(int) + (m + 1) * sizeof(int) 
            + k * n * sizeof(double) / ngpu 
            + m * n * sizeof(double) / ngpu);

    if ( matrix_size_in_gb > get_gpu_availble_mem(ngpu)) {
        cout << "No available device memory for " << matrix_size_in_gb << " GB \n";
        return -1;
    }

    for (int d = 0; d < ngpu; d++) {
        dev_n[d] = floor((d + 1) * n / ngpu) - floor(d * n / ngpu);
        B_offset[d] = floor(d * n / ngpu) * k;
        C_offset[d] = floor(d * n / ngpu) * m;
        cuda_status[d] = new cudaError_t[5];
        cusparse_status[d] = new cusparseStatus_t[2];
    }

    time_parse = get_time() - curr_time;
	cout << "parse time for " << ngpu << " GPUs: " << time_parse << endl;

	curr_time = get_time();

	for (int d = 0; d < ngpu; d++) {
		cudaSetDevice(d);

		cudaStreamCreate(&(stream[d]));
		
		cusparse_status[d][0] = cusparseCreate(&(handle[d])); 
		if (cusparse_status[d][0] != CUSPARSE_STATUS_SUCCESS) { 
			printf("CUSPARSE Library initialization failed\n");
			return 1; 
		} 
		cusparse_status[d][0] = cusparseSetStream(handle[d], stream[d]);
		if (cusparse_status[d][0] != CUSPARSE_STATUS_SUCCESS) {
			printf("Stream bindind failed\n");
			return 1;
		} 
		cusparse_status[d][0] = cusparseCreateMatDescr(&mat_A[d]);
		if (cusparse_status[d][0] != CUSPARSE_STATUS_SUCCESS) { 
			printf("Matrix descriptor initialization failed\n");
			return 1;
		} 	
		cusparseSetMatType(mat_A[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(mat_A[d],CUSPARSE_INDEX_BASE_ZERO); 
	}

        for (int d = 0; d < ngpu; d++) {
/*
                cudaSetDevice(d);
		cuda_status[d][0] = cudaMalloc((void**)&dev_csrRowPtr_A[d],
					       (m + 1) * sizeof(int));
		cuda_status[d][1] = cudaMalloc((void**)&dev_csrColIndex_A[d],
					       nnz_A * sizeof(int)); 
		cuda_status[d][2] = cudaMalloc((void**)&dev_csrVal_A[d],
					       nnz_A * sizeof(double)); 
		cuda_status[d][3] = cudaMalloc((void**)&dev_B_dense[d],
					       k * dev_n[d] * sizeof(double)); 
		cuda_status[d][4] = cudaMalloc((void**)&dev_C_dense[d],
					       m * dev_n[d] * sizeof(double)); 
		
		for (int i = 0; i < 5; ++i) {
			if (cuda_status[d][i] != cudaSuccess) {
				cout << "cudaMalloc for device " << d << " failed! \n"; 
				return 1;
			}
		}
*/
                cudaSetDevice(d);
                cudaMalloc((void**)&dev_csrRowPtr_A[d], (m + 1) * sizeof(int));
                cudaMalloc((void**)&dev_csrColIndex_A[d], nnz_A * sizeof(int));
                cudaMalloc((void**)&dev_csrVal_A[d], nnz_A * sizeof(double));
                cudaMalloc((void**)&dev_B_dense[d], k * dev_n[d] * sizeof(double));
                cudaMalloc((void**)&dev_C_dense[d], m * dev_n[d] * sizeof(double));
        }

        for (int d = 0; d < ngpu; d++) {
                cudaSetDevice(d);
		//cout << "Start copy to GPUs...";
		cudaMemcpyAsync(dev_csrRowPtr_A[d],
				host_csrRowPtr_A,
				(size_t)((m + 1) * sizeof(int)),
				cudaMemcpyHostToDevice,
				stream[d]);
		
		cudaMemcpyAsync(dev_csrColIndex_A[d],
				host_csrColIndex_A,
				(size_t)(nnz_A * sizeof(int)),
				cudaMemcpyHostToDevice,
                                stream[d]); 
		
		cudaMemcpyAsync(dev_csrVal_A[d],
				host_csrVal_A,
				(size_t)(nnz_A * sizeof(double)),
				cudaMemcpyHostToDevice,
                                stream[d]);

		cudaMemcpyAsync(dev_B_dense[d],
				host_B_dense + (size_t)B_offset[d],
				(size_t)(k * dev_n[d] * sizeof(double)),
				cudaMemcpyHostToDevice,
                                stream[d]);
 
		cudaMemcpyAsync(dev_C_dense[d],
				host_C_dense + (size_t)C_offset[d],
				(size_t)(m * dev_n[d] * sizeof(double)),
				cudaMemcpyHostToDevice,
                                stream[d]);
	}

        for (int d = 0; d < ngpu; d++) {
                cudaSetDevice(d);
		cudaDeviceSynchronize();
        }


	time_comm = get_time() - curr_time;
	cout << "comm time for " << ngpu << " GPUs: " << time_comm << endl;

	curr_time = get_time();
	for (int d = 0; d < ngpu; ++d) {
		cudaSetDevice(d);
		cusparse_status[d][0] = cusparseDcsrmm(handle[d],
						       CUSPARSE_OPERATION_NON_TRANSPOSE, 
						       m,
						       dev_n[d],
						       k,
						       nnz_A,
						       alpha,
						       mat_A[d],
						       dev_csrVal_A[d],
						       dev_csrRowPtr_A[d],
						       dev_csrColIndex_A[d],
						       dev_B_dense[d],
						       k,
						       beta,
						       dev_C_dense[d],
						       m);		 	 	
	}

	for (int d = 0; d < ngpu; ++d) {
		cudaSetDevice(d);
		cudaDeviceSynchronize();
		if (cusparse_status[d][0] != CUSPARSE_STATUS_SUCCESS) {
			cout << "cusparseDcsrmm for device " << d << " failed! \n";
			return -1;
		}
	}

	time_comp = get_time() - curr_time;
	cout << "comp time for " << ngpu << " GPUs: " << time_comp << endl;

	curr_time = get_time();

	for (int d = 0; d < ngpu; d++) {
                cudaSetDevice(d);
		cudaMemcpyAsync(host_C_dense + (size_t)C_offset[d],
				dev_C_dense[d],
				(size_t)(m * dev_n[d] * sizeof(double)),
				cudaMemcpyHostToDevice,
				stream[d]);
	}

	for (int d = 0; d < ngpu; d++) {
		cudaSetDevice(d);
		cudaDeviceSynchronize();
		cusparseDestroyMatDescr(mat_A[d]);
		cusparseDestroy(handle[d]);
		cudaFree(dev_csrVal_A[d]);
		cudaFree(dev_csrRowPtr_A[d]);
		cudaFree(dev_csrColIndex_A[d]);
		cudaFree(dev_B_dense[d]);
		cudaFree(dev_C_dense[d]);
	}
	
	for (int d = 0; d < ngpu; d++) {
		delete [] cuda_status[d];
		delete [] cusparse_status[d];
	}
	
	delete [] dev_n;
	delete [] cusparse_status;
	delete [] cuda_status;
	delete [] dev_C_dense;
	delete [] dev_B_dense;
	delete [] dev_csrVal_A;
	delete [] dev_csrColIndex_A;
	delete [] dev_csrRowPtr_A;
	delete [] mat_A;
	delete [] handle;
	delete [] stream;

	time_post = get_time() - curr_time;
	cout << "post time for " << ngpu << " GPUs: " << time_post << endl;

	return 0;
}
