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


int spMV_mgpu_v1(int m, int n, long long nnz, double * alpha,
				  double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel){

		double curr_time = 0.0;
		double time_parse = 0.0;
		double time_comm = 0.0;
		double time_comp = 0.0;
		double time_post = 0.0;


		curr_time = get_time();


		long long  * start_idx  = new long long[ngpu];
		long long  * end_idx    = new long long[ngpu];
		int        * start_row  = new int[ngpu];
		int        * end_row    = new int[ngpu];
		bool       * start_flag = new bool[ngpu];
		bool       * end_flag   = new bool[ngpu];

		double ** dev_csrVal      = new double * [ngpu];
		int    ** host_csrRowPtr  = new int    * [ngpu];
		int    ** dev_csrRowPtr   = new int    * [ngpu];
		int    ** dev_csrColIndex = new int    * [ngpu];
		int    *         dev_nnz  = new int      [ngpu];
		int    *           dev_m  = new int      [ngpu];
		int    *           dev_n  = new int      [ngpu];

		double ** dev_x  = new double * [ngpu];
		double ** dev_y  = new double * [ngpu];
		double ** host_y = new double * [ngpu];
		double *  y2     = new double   [ngpu];

		cudaStream_t       * stream = new cudaStream_t [ngpu];
		cusparseStatus_t   * status = new cusparseStatus_t[ngpu];
		cusparseHandle_t   * handle = new cusparseHandle_t[ngpu];
		cusparseMatDescr_t * descr  = new cusparseMatDescr_t[ngpu];
		int * err = new int[ngpu];

		// Calculate the start and end index
		for (int i = 0; i < ngpu; i++) {

			long long tmp1 = i * nnz;
			long long tmp2 = (i + 1) * nnz;

			double tmp3 = (double)(tmp1 / ngpu);
			double tmp4 = (double)(tmp2 / ngpu);

			start_idx[i] = floor((double)tmp1 / ngpu);
			end_idx[i]   = floor((double)tmp2 / ngpu) - 1;
		}

		// Calculate the start and end row
		for (int i = 0; i < ngpu; i++) {
			start_row[i] = get_row_from_index(m, csrRowPtr, start_idx[i]);
			// Mark imcomplete rows
			// True: imcomplete
			if (start_idx[i] > csrRowPtr[start_row[i]]) {
				start_flag[i] = true;
				y2[i] = y[start_row[i]];
			} else {
				start_flag[i] = false;
			}
		}

		for (int i = 0; i < ngpu; i++) {
			end_row[i] = get_row_from_index(m, csrRowPtr, end_idx[i]);
			// Mark imcomplete rows
			// True: imcomplete
			if (end_idx[i] < csrRowPtr[end_row[i] + 1] - 1)  {
				end_flag[i] = true;
			} else {
				end_flag[i] = false;
			}
		}

		// Cacluclate dimensions
		for (int i = 0; i < ngpu; i++) {
			dev_m[i] = end_row[i] - start_row[i] + 1;
			dev_n[i] = n;
		}

		for (int i = 0; i < ngpu; i++) {
			host_y[i] = new double[dev_m[i]];
		}

		for (int d = 0; d < ngpu; d++) {
			long long nnz_ll = end_idx[d] - start_idx[d] + 1;
			long long matrix_data_space = nnz_ll * sizeof(double) + 
										nnz_ll * sizeof(int) + 
										(long long)(dev_m[d]+1) * sizeof(int) + 
										(long long)dev_n[d] * sizeof(double) +
										(long long)dev_m[d] * sizeof(double);
			double matrix_size_in_gb = (double)matrix_data_space / 1e9;
			if ( matrix_size_in_gb > 0.8 * get_gpu_availble_mem(ngpu)) {
				return -1;
			}


			dev_nnz[d]   = (int)(end_idx[d] - start_idx[d] + 1);
		}


		

		for (int i = 0; i < ngpu; i++) {
			host_csrRowPtr[i] = new int [dev_m[i] + 1];
			host_csrRowPtr[i][0] = 0;
			host_csrRowPtr[i][dev_m[i]] = dev_nnz[i];

			for (int j = 1; j < dev_m[i]; j++) {
				host_csrRowPtr[i][j] = (int)(csrRowPtr[start_row[i] + j] - start_idx[i]);
			}
		}

		for (int d = 0; d < ngpu; d++) {

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
		}

		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d]     * sizeof(double));
			cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int)   );
			cudaMalloc((void**)&dev_csrColIndex[d], dev_nnz[d]     * sizeof(int)   );
			cudaMalloc((void**)&dev_x[d],           dev_n[d]       * sizeof(double)); 
		    cudaMalloc((void**)&dev_y[d],           dev_m[d]       * sizeof(double)); 
		}


		time_parse = get_time() - curr_time;

		curr_time = get_time();

		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaMemcpyAsync(dev_csrRowPtr[d],   host_csrRowPtr[d],          (size_t)((dev_m[d] + 1) * sizeof(int)), cudaMemcpyHostToDevice, stream[d]);
			cudaMemcpyAsync(dev_csrColIndex[d], &csrColIndex[start_idx[d]], (size_t)(dev_nnz[d] * sizeof(int)),     cudaMemcpyHostToDevice, stream[d]); 
			cudaMemcpyAsync(dev_csrVal[d],      &csrVal[start_idx[d]],      (size_t)(dev_nnz[d] * sizeof(double)),  cudaMemcpyHostToDevice, stream[d]); 

			cudaMemcpyAsync(dev_y[d], &y[start_row[d]], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyHostToDevice, stream[d]); 
			cudaMemcpyAsync(dev_x[d], x,                (size_t)(dev_n[d]*sizeof(double)),  cudaMemcpyHostToDevice, stream[d]); 
		}

		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
		}
		time_comm = get_time() - curr_time;
		curr_time = get_time();


		for (int d = 0; d < ngpu; ++d) 
		{
			err[d] = 0;
			cudaSetDevice(d);
			if (kernel == 1) {
				status[d] = cusparseDcsrmv(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											dev_m[d], dev_n[d], dev_nnz[d], 
											alpha, descr[d], dev_csrVal[d], 
											dev_csrRowPtr[d], dev_csrColIndex[d], 
											dev_x[d],  beta, dev_y[d]); 
			} else if (kernel == 2) {
				status[d] = cusparseDcsrmv_mp(handle[d],CUSPARSE_OPERATION_NON_TRANSPOSE, 
											dev_m[d], dev_n[d], dev_nnz[d], 
											alpha, descr[d], dev_csrVal[d], 
											dev_csrRowPtr[d], dev_csrColIndex[d], 
											dev_x[d],  beta, dev_y[d]); 
			}
           /* 
            else if (kernel == 3) {
				err[d] = csr5_kernel(dev_m[d], dev_n[d], dev_nnz[d], 
							alpha, dev_csrVal[d], 
							dev_csrRowPtr[d], dev_csrColIndex[d], 
							dev_x[d],  beta, dev_y[d]); 
			}
            */
		}

		for (int d = 0; d < ngpu; ++d) 
		{
			cudaSetDevice(d);
			cudaDeviceSynchronize();
			if (status[d] != CUSPARSE_STATUS_SUCCESS || err[d] != 0 ) {
				return -1;
			}

		}

		time_comp = get_time() - curr_time;
		curr_time = get_time();

		for (int d = 0; d < ngpu; d++) {
			double tmp = 0.0;
			
			if (start_flag[d]) {
				tmp = y[start_row[d]];
			}
	
			cudaMemcpy(&y[start_row[d]], dev_y[d], (size_t)(dev_m[d]*sizeof(double)),  cudaMemcpyDeviceToHost); 

			if (start_flag[d]) {
				y[start_row[d]] += tmp;
				y[start_row[d]] -= y2[d] * (*beta);
			}
		}
		for (int d = 0; d < ngpu; d++) {
			cudaSetDevice(d);
			cudaFree(dev_csrVal[d]);
			cudaFree(dev_csrRowPtr[d]);
			cudaFree(dev_csrColIndex[d]);
			cudaFree(dev_x[d]);
			cudaFree(dev_y[d]);
			delete [] host_y[d];
			delete [] host_csrRowPtr[d];
			cusparseDestroyMatDescr(descr[d]);
			cusparseDestroy(handle[d]);
			cudaStreamDestroy(stream[d]);

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

