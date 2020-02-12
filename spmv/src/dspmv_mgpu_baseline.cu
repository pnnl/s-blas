#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
//#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"
using namespace std;

int spMV_mgpu_baseline(int m, int n, long long nnz, double * alpha,
				 		double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				 		double * x, double * beta,
				 		double * y,
				 		int ngpu){

	double curr_time = 0.0;
	double time_parse = 0.0;
	double time_comm = 0.0;
	double time_comp = 0.0;
	double time_post = 0.0;


	curr_time = get_time();

	cudaStream_t * stream = new cudaStream_t [ngpu];

	cudaError_t * cudaStat1 = new cudaError_t[ngpu];
	cudaError_t * cudaStat2 = new cudaError_t[ngpu];
	cudaError_t * cudaStat3 = new cudaError_t[ngpu];
	cudaError_t * cudaStat4 = new cudaError_t[ngpu];
	cudaError_t * cudaStat5 = new cudaError_t[ngpu];
	cudaError_t * cudaStat6 = new cudaError_t[ngpu];

	cusparseStatus_t * status = new cusparseStatus_t[ngpu];
	cusparseHandle_t * handle = new cusparseHandle_t[ngpu];
	cusparseMatDescr_t * descr = new cusparseMatDescr_t[ngpu];

	int  * start_row  = new int[ngpu];
	int  * end_row    = new int[ngpu];
		
	int * dev_m            = new int      [ngpu];
	int * dev_n            = new int      [ngpu];
	int * dev_nnz          = new int      [ngpu];
	int ** host_csrRowPtr  = new int    * [ngpu];
	int ** dev_csrRowPtr   = new int    * [ngpu];
	int ** dev_csrColIndex = new int    * [ngpu];
	double ** dev_csrVal   = new double * [ngpu];


	double ** dev_x = new double * [ngpu];
	double ** dev_y = new double * [ngpu];

	


	for (int d = 0; d < ngpu; d++){

		cudaSetDevice(d);

		start_row[d] = floor((d)     * m / ngpu);
		end_row[d]   = floor((d + 1) * m / ngpu) - 1;

		dev_m[d]   = end_row[d] - start_row[d] + 1;
		dev_n[d]   = n;

		long long nnz_ll = csrRowPtr[end_row[d] + 1] - csrRowPtr[start_row[d]];
		long long matrix_data_space = nnz_ll * sizeof(double) + 
										nnz_ll * sizeof(int) + 
										(long long)(dev_m[d]+1) * sizeof(int) + 
										(long long)dev_n[d] * sizeof(double) +
										(long long)dev_m[d] * sizeof(double);
		double matrix_size_in_gb = (double)matrix_data_space / 1e9;
		if ( matrix_size_in_gb > 0.8 * get_gpu_availble_mem(ngpu)) {
			return -1;
		}

		dev_nnz[d] = (int)(csrRowPtr[end_row[d] + 1] - csrRowPtr[start_row[d]]);
		host_csrRowPtr[d] = new int[dev_m[d] + 1];
		for (int i = 0; i < dev_m[d] + 1; i++) {
			host_csrRowPtr[d][i] = (int)(csrRowPtr[start_row[d] + i] - csrRowPtr[start_row[d]]);
		}

	}


	time_parse = get_time() - curr_time;
	curr_time = get_time();

	for (int d = 0; d < ngpu; d++){
		cudaSetDevice(d);

		cudaStreamCreate(&(stream[d]));
		
		status[d] = cusparseCreate(&(handle[d])); 
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("CUSPARSE Library initialization failed");
			return 1; 
		} 
		status[d] = cusparseSetStream(handle[d], stream[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("Stream bindind failed");
			return 1;
		} 
		status[d] = cusparseCreateMatDescr(&descr[d]);
		if (status[d] != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("Matrix descriptor initialization failed");
			return 1;
		} 	
		cusparseSetMatType(descr[d],CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(descr[d],CUSPARSE_INDEX_BASE_ZERO); 

		cudaStat1[d] = cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int));
		cudaStat2[d] = cudaMalloc((void**)&dev_csrColIndex[d], dev_nnz[d] * sizeof(int)); 
		cudaStat3[d] = cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d] * sizeof(double)); 

		cudaStat4[d] = cudaMalloc((void**)&dev_x[d],           dev_n[d] * sizeof(double)); 
		cudaStat5[d] = cudaMalloc((void**)&dev_y[d],           dev_m[d] * sizeof(double)); 
		

		if ((cudaStat1[d] != cudaSuccess) || 
			(cudaStat2[d] != cudaSuccess) || 
			(cudaStat3[d] != cudaSuccess) || 
			(cudaStat4[d] != cudaSuccess) || 
			(cudaStat5[d] != cudaSuccess)) 
		{ 
			printf("Device malloc failed");
			return 1; 
		} 

		//cout << "Start copy to GPUs...";
		cudaStat1[d] = cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],                  (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice);
		cudaStat2[d] = cudaMemcpy(dev_csrColIndex[d], &csrColIndex[csrRowPtr[start_row[d]]], (size_t)(dev_nnz[d] * sizeof(int)),   cudaMemcpyHostToDevice); 
		cudaStat3[d] = cudaMemcpy(dev_csrVal[d],      &csrVal[csrRowPtr[start_row[d]]],      (size_t)(dev_nnz[d] * sizeof(double)), cudaMemcpyHostToDevice);
		cudaStat4[d] = cudaMemcpy(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)), cudaMemcpyHostToDevice); 
		cudaStat5[d] = cudaMemcpy(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)), cudaMemcpyHostToDevice); 
		

		if ((cudaStat1[d] != cudaSuccess) ||
		 	(cudaStat2[d] != cudaSuccess) ||
		  	(cudaStat3[d] != cudaSuccess) ||
		   	(cudaStat4[d] != cudaSuccess) ||
		    (cudaStat5[d] != cudaSuccess)) 
		{ 
			printf("Memcpy from Host to Device failed"); 
			return 1; 
		} 

	}

	time_comm = get_time() - curr_time;
	curr_time = get_time();

	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
								   dev_m[d], dev_n[d], dev_nnz[d], 
								   alpha, descr[d], dev_csrVal[d], 
								   dev_csrRowPtr[d], dev_csrColIndex[d], 
								   dev_x[d], beta, dev_y[d]);		 	 	
	}
	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
		if (status[d] != CUSPARSE_STATUS_SUCCESS) {
			return -1;
		}
	}


	

	time_comp = get_time() - curr_time;
	curr_time = get_time();

	for (int d = 0; d < ngpu; d++)
	{
		cudaMemcpy( &y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost);
	}

	for (int d = 0; d < ngpu; d++) {
		cudaSetDevice(d);
		cudaFree(dev_csrVal[d]);
		cudaFree(dev_csrRowPtr[d]);
		cudaFree(dev_csrColIndex[d]);
		cudaFree(dev_x[d]);
		cudaFree(dev_y[d]);
	}

	
	delete[] dev_csrVal;
	delete[] dev_csrRowPtr;
	delete[] dev_csrColIndex;
	delete[] dev_x;
	delete[] dev_y;
	delete[] host_csrRowPtr;
	delete[] start_row;
	delete[] end_row;

	time_post = get_time() - curr_time;
		
	//cout << "time_parse = " << time_parse << ", time_comm = " << time_comm << ", time_comp = "<< time_comp <<", time_post = " << time_post << endl;

	return 0;

}
