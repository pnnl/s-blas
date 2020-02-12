#include <cuda_runtime.h>
#include "cusparse.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <pthread.h>
#include "spmv_task.h"
#include "spmv_kernel.h"
#include <omp.h>
//#include "anonymouslib_cuda.h"

using namespace std;

void * spmv_worker(void * arg);

void generate_tasks(int m, int n, long long nnz, double * alpha,
				    double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  	double * x, double * beta,
				  	double * y,
				  	long long nb,
				  	vector<spmv_task *> * spmv_task_pool_ptr);

void assign_task(spmv_task * t, int dev_id, cudaStream_t stream);

int run_task(spmv_task * t, int dev_id, cusparseHandle_t handle, int kernel);

void finalize_task(spmv_task * t, int dev_id, cudaStream_t stream);

void gather_results(vector<spmv_task *> * spmv_task_completed, double * y, double * beta, int m);

void print_task_info(spmv_task * t);

int spMV_mgpu_v2(int m, int n, long long nnz, double * alpha,
				  double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel,
				  long long nb,
				  int q)
{

	nb = min(nb, (long long )(0.8*get_gpu_availble_mem(ngpu)*1e9/(double)(sizeof(double) + sizeof(int) + sizeof(int)))/q ); 
	if (nb <= 0 || ngpu == 0 || q == 0) {
		return -1;
	}


	double curr_time = 0.0;
	double time_parse = 0.0;
	double time_comm_comp = 0.0;
	double time_post = 0.0;

	curr_time = get_time();

	vector<spmv_task *> * spmv_task_pool = new vector<spmv_task *>();
	vector<spmv_task *> * spmv_task_completed = new vector<spmv_task *>();

	generate_tasks(m, n, nnz, alpha,
				  csrVal, csrRowPtr, csrColIndex, 
				  x, beta, y, nb,
				  spmv_task_pool);

	int num_of_tasks = (*spmv_task_pool).size();

	(*spmv_task_completed).reserve(num_of_tasks);

	time_parse = get_time() - curr_time;

	curr_time = get_time();

	//cout << "starting " << ngpu << " GPUs." << endl;
	omp_set_num_threads(ngpu);
	//cout << "omp_get_max_threads = " << omp_get_max_threads() << endl;
	//cout << "omp_get_thread_limit = " << omp_get_thread_limit() << endl;
	#pragma omp parallel default (shared)
	{
		
		int c;
		unsigned int dev_id = omp_get_thread_num();
		//cout << "thread " << dev_id <<"/" << omp_get_num_threads() << "started" << endl;
		cudaSetDevice(dev_id);
		
		cusparseStatus_t status[q];
		cudaStream_t stream[q];
		cusparseHandle_t handle[q];



		double ** dev_csrVal = new double * [q];
		int ** dev_csrRowPtr = new int    * [q];
		int ** dev_csrColIndex = new int  * [q];
		double ** dev_x = new double      * [q];
		double ** dev_y = new double      * [q];

		for (c = 0; c < q; c++) {
			cudaStreamCreate(&(stream[c]));
			status[c] = cusparseCreate(&(handle[c])); 
			if (status[c] != CUSPARSE_STATUS_SUCCESS) 
			{ 
				printf("CUSPARSE Library initialization failed");
				//return 1; 
			} 
			status[c] = cusparseSetStream(handle[c], stream[c]);
			if (status[c] != CUSPARSE_STATUS_SUCCESS) 
			{ 
				printf("Stream bindind failed");
				//return 1;
			} 

			cudaMalloc((void**)&(dev_csrVal[c]),      nb      * sizeof(double));
			cudaMalloc((void**)&(dev_csrRowPtr[c]),   (m + 1) * sizeof(int)   );
			cudaMalloc((void**)&(dev_csrColIndex[c]), nb      * sizeof(int)   );
			cudaMalloc((void**)&(dev_x[c]),           n       * sizeof(double));
	    	cudaMalloc((void**)&(dev_y[c]),           m       * sizeof(double));

    	}

   		c = 0; 
    	
    	//cout << "GPU " << dev_id << " entering loop" << endl;


    	int num_of_assigned_task = 0;
    	int num_of_to_be_assigned_task = num_of_tasks * (dev_id + 1) /  omp_get_num_threads() - 
    									 num_of_tasks * (dev_id) /  omp_get_num_threads();

		while (true) {

			spmv_task * curr_spmv_task;

			for (c = 0; c < q; c++) {

				//cout << "GPU " << dev_id << " try to get one task" << endl;
				#pragma omp critical
				{

					if(num_of_assigned_task < num_of_to_be_assigned_task &&
						 (*spmv_task_pool).size() > 0) {
						curr_spmv_task = (*spmv_task_pool)[(*spmv_task_pool).size() - 1];
						(*spmv_task_pool).pop_back();
						(*spmv_task_completed).push_back(curr_spmv_task);
						num_of_assigned_task++;
						//cout << "GPU " << dev_id << " got one task" << endl;
						//cout << "Number of task left: " << (*spmv_task_pool).size() << "/" << num_of_tasks << endl;
					} else {
						curr_spmv_task = NULL;
					}
				}

				if (curr_spmv_task) {

					curr_spmv_task->dev_csrVal = dev_csrVal[c];
					curr_spmv_task->dev_csrRowPtr = dev_csrRowPtr[c];
					curr_spmv_task->dev_csrColIndex = dev_csrColIndex[c];
					curr_spmv_task->dev_x = dev_x[c];
					curr_spmv_task->dev_y = dev_y[c];
					assign_task(curr_spmv_task, dev_id, stream[c]);
					run_task(curr_spmv_task, dev_id, handle[c], kernel);
					finalize_task(curr_spmv_task, dev_id, stream[c]);
				}
				if (!curr_spmv_task) {
					break;
				}

			}
			if (!curr_spmv_task) {
				break;
			}
		}

		cudaDeviceSynchronize();

		for (c = 0; c < q; c++) {

			cudaFree(dev_csrVal[c]);
			cudaFree(dev_csrRowPtr[c]);
			cudaFree(dev_csrColIndex[c]);
			cudaFree(dev_x[c]);
			cudaFree(dev_y[c]);
			cusparseDestroy(handle[c]);
			cudaStreamDestroy(stream[c]);
		}

		


	}

	time_comm_comp = get_time() - curr_time;

	curr_time = get_time();

	gather_results(spmv_task_completed, y, beta, m);

	for (int t = 0; t < (*spmv_task_completed).size(); t++) {
		cudaFreeHost((*spmv_task_completed)[t]->host_csrRowPtr);
		cudaFreeHost((*spmv_task_completed)[t]->local_result_y);
		cudaFreeHost((*spmv_task_completed)[t]->alpha);
		cudaFreeHost((*spmv_task_completed)[t]->beta);

	}

	time_post = get_time() - curr_time;

	//cout << "time_parse = " << time_parse << ", time_comm_comp = " << time_comm_comp << ", time_post = " << time_post << endl;
}



void generate_tasks(int m, int n, long long nnz, double * alpha,
				    double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  	double * x, double * beta,
				  	double * y,
				  	long long nb,
				  	vector<spmv_task *> * spmv_task_pool_ptr) {

	int num_of_tasks = (int)((nnz + nb - 1) / nb);
	//cout << "num_of_tasks = " << num_of_tasks << endl;

	int curr_row;
	int t;
	int d;

	spmv_task * spmv_task_pool = new spmv_task[num_of_tasks];

	// Calculate the start and end index
	for (t = 0; t < num_of_tasks; t++) {
		long long tmp1 = t * nnz;
		long long tmp2 = (t + 1) * nnz;

		double tmp3 = (double)(tmp1 / num_of_tasks);
		double tmp4 = (double)(tmp2 / num_of_tasks);

		spmv_task_pool[t].start_idx = floor((double)(tmp1 / num_of_tasks));
		spmv_task_pool[t].end_idx   = floor((double)(tmp2 / num_of_tasks)) - 1;
		spmv_task_pool[t].dev_nnz = (int)(spmv_task_pool[t].end_idx - spmv_task_pool[t].start_idx + 1);
	}

	// Calculate the start and end row
	curr_row = 0;
	for (t = 0; t < num_of_tasks; t++) {

		spmv_task_pool[t].start_row = get_row_from_index(m, csrRowPtr, spmv_task_pool[t].start_idx);
		// Mark imcomplete rows
		// True: imcomplete
		if (spmv_task_pool[t].start_idx > csrRowPtr[spmv_task_pool[t].start_row]) {
			spmv_task_pool[t].start_flag = true;
			spmv_task_pool[t].y2 = y[spmv_task_pool[t].start_row];
		} else {
			spmv_task_pool[t].start_flag = false;
		}
	}

	curr_row = 0;
	for (t = 0; t < num_of_tasks; t++) {
		spmv_task_pool[t].end_row = get_row_from_index(m, csrRowPtr, spmv_task_pool[t].end_idx);

		// Mark imcomplete rows
		// True: imcomplete
		if (spmv_task_pool[t].end_idx < csrRowPtr[spmv_task_pool[t].end_row + 1] - 1)  {
			spmv_task_pool[t].end_flag = true;
			spmv_task_pool[t].y2 = y[spmv_task_pool[t].end_row];
		} else {
			spmv_task_pool[t].end_flag = false;
		}
	}

	// Cacluclate dimensions
	for (t = 0; t < num_of_tasks; t++) {
		spmv_task_pool[t].dev_m = spmv_task_pool[t].end_row - spmv_task_pool[t].start_row + 1;
		spmv_task_pool[t].dev_n = n;
	}


	for (t = 0; t < num_of_tasks; t++) {
		cudaMallocHost((void **)&(spmv_task_pool[t].host_csrRowPtr), (spmv_task_pool[t].dev_m + 1) * sizeof(int));

		spmv_task_pool[t].host_csrRowPtr[0] = 0;
		spmv_task_pool[t].host_csrRowPtr[spmv_task_pool[t].dev_m] = spmv_task_pool[t].dev_nnz;
	
		// memcpy(&(spmv_task_pool[t].host_csrRowPtr[1]), 
		// 	   &csrRowPtr[spmv_task_pool[t].start_row + 1], 
		// 	   (spmv_task_pool[t].dev_m - 1) * sizeof(int) );

	
		for (int j = 1; j < spmv_task_pool[t].dev_m; j++) {
			spmv_task_pool[t].host_csrRowPtr[j] = (int)(csrRowPtr[spmv_task_pool[t].start_row + j] - spmv_task_pool[t].start_idx);
		}

		spmv_task_pool[t].host_csrColIndex = csrColIndex;
		spmv_task_pool[t].host_csrVal = csrVal;
		spmv_task_pool[t].host_y = y;
		spmv_task_pool[t].host_x = x;

		cudaMallocHost((void **)&(spmv_task_pool[t].local_result_y), spmv_task_pool[t].dev_m * sizeof(double));

		cudaMallocHost((void **)&(spmv_task_pool[t].alpha), 1 * sizeof(double));

		cudaMallocHost((void **)&(spmv_task_pool[t].beta), 1 * sizeof(double));

		spmv_task_pool[t].alpha[0] = *alpha;
		spmv_task_pool[t].beta[0] = *beta;
	}

	for (t = 0; t < num_of_tasks; t++) {
		cusparseStatus_t status = cusparseCreateMatDescr(&(spmv_task_pool[t].descr));
		if (status != CUSPARSE_STATUS_SUCCESS) 
		{ 
			printf("Matrix descriptor initialization failed");
			//return 1;
		} 	
		cusparseSetMatType(spmv_task_pool[t].descr,CUSPARSE_MATRIX_TYPE_GENERAL); 
		cusparseSetMatIndexBase(spmv_task_pool[t].descr,CUSPARSE_INDEX_BASE_ZERO);
	}

	(*spmv_task_pool_ptr).reserve(num_of_tasks);
	for (t = 0; t < num_of_tasks; t++) {
		(*spmv_task_pool_ptr).push_back(&spmv_task_pool[t]);
	}

}

void assign_task(spmv_task * t, int dev_id, cudaStream_t stream){
	t->dev_id = dev_id;
	cudaSetDevice(dev_id);

    cudaMemcpyAsync(t->dev_csrRowPtr,   t->host_csrRowPtr,          
    			   (size_t)((t->dev_m + 1) * sizeof(int)), cudaMemcpyHostToDevice, stream);

	cudaMemcpyAsync(t->dev_csrColIndex, &(t->host_csrColIndex[t->start_idx]), 
		           (size_t)(t->dev_nnz * sizeof(int)), cudaMemcpyHostToDevice, stream); 

	cudaMemcpyAsync(t->dev_csrVal,      &(t->host_csrVal[t->start_idx]),
		           (size_t)(t->dev_nnz * sizeof(double)), cudaMemcpyHostToDevice, stream); 

	cudaMemcpyAsync(t->dev_y, &(t->host_y[t->start_row]), 
		           (size_t)(t->dev_m * sizeof(double)), cudaMemcpyHostToDevice, stream); 
	
	cudaMemcpyAsync(t->dev_x, t->host_x,
				   (size_t)(t->dev_n * sizeof(double)),  cudaMemcpyHostToDevice, stream);

}

int run_task(spmv_task * t, int dev_id, cusparseHandle_t handle, int kernel){
	cudaSetDevice(dev_id);

	cusparseStatus_t status;
	int err;
	if(kernel == 1) {
		status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
								t->dev_m, t->dev_n, t->dev_nnz, 
								t->alpha, t->descr, t->dev_csrVal, 
								t->dev_csrRowPtr, t->dev_csrColIndex, 
								t->dev_x,  t->beta, t->dev_y); 
	} else if (kernel == 2) {
		status = cusparseDcsrmv_mp(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
									t->dev_m, t->dev_n, t->dev_nnz, 
									t->alpha, t->descr, t->dev_csrVal, 
									t->dev_csrRowPtr, t->dev_csrColIndex, 
									t->dev_x,  t->beta, t->dev_y); 
	} 
    
    /*
    else if (kernel == 3) {
		err = csr5_kernel(t->dev_m, t->dev_n, t->dev_nnz, 
					t->alpha, t->dev_csrVal, 
					t->dev_csrRowPtr, t->dev_csrColIndex, 
					t->dev_x,  t->beta, t->dev_y); 
		
	}
    */
	if (status != CUSPARSE_STATUS_SUCCESS || err != 0 ) {
		return -1;
	}
}

void finalize_task(spmv_task * t, int dev_id, cudaStream_t stream) {
	cudaSetDevice(dev_id);
	cudaMemcpyAsync(t->local_result_y,   t->dev_y,          
    			   (size_t)((t->dev_m) * sizeof(double)), 
    			   cudaMemcpyDeviceToHost, stream);
}

void gather_results(vector<spmv_task *> * spmv_task_completed, double * y, double * beta, int m) {
	
	int t = 0;
	bool * flag = new bool[m];
	for (int i = 0; i < m; i++) {
		flag[i] = false;
	}
	for (t = 0; t < (*spmv_task_completed).size(); t++) {
		 //cout << "Task " << t << endl;
		 //cout << "flag = " << (*spmv_task_completed)[t]->start_flag <<" " <<   (*spmv_task_completed)[t]->end_flag << endl;
		 //for (int i = 0; i < (*spmv_task_completed)[t]->dev_m; i++) {
		 //	cout << (*spmv_task_completed)[t]->local_result_y[i] << " ";
		 //}
		 //cout << endl;

		double tmp = 0.0;

		if ((*spmv_task_completed)[t]->dev_m == 1 && 
			((*spmv_task_completed)[t]->start_flag) && 
			((*spmv_task_completed)[t]->end_flag)) {
				if (!flag[(*spmv_task_completed)[t]->start_row]) {
					flag[(*spmv_task_completed)[t]->start_row] = true;
				} else {
					tmp = y[(*spmv_task_completed)[t]->start_row];
					(*spmv_task_completed)[t]->local_result_y[0] += tmp;
					(*spmv_task_completed)[t]->local_result_y[0] -= (*beta) * (*spmv_task_completed)[t]->y2;
				}
		}
		
		else {
			if ((*spmv_task_completed)[t]->start_flag) {
				if (!flag[(*spmv_task_completed)[t]->start_row]) {
					flag[(*spmv_task_completed)[t]->start_row] = true;
				} else {
					tmp = y[(*spmv_task_completed)[t]->start_row];
					(*spmv_task_completed)[t]->local_result_y[0] += tmp;
					(*spmv_task_completed)[t]->local_result_y[0] -= (*beta) * (*spmv_task_completed)[t]->y2;
				}
			}

			if ((*spmv_task_completed)[t]->end_flag) {
				if (!flag[(*spmv_task_completed)[t]->end_row]) {
					flag[(*spmv_task_completed)[t]->end_row] = true;
				} else {
					tmp = y[(*spmv_task_completed)[t]->end_row];
					(*spmv_task_completed)[t]->local_result_y[(*spmv_task_completed)[t]->dev_m - 1] += tmp;
					(*spmv_task_completed)[t]->local_result_y[(*spmv_task_completed)[t]->dev_m - 1] -= (*beta) * (*spmv_task_completed)[t]->y2;
				}
			}
		}

		memcpy(&y[(*spmv_task_completed)[t]->start_row], 
			   (*spmv_task_completed)[t]->local_result_y, 
			  ((*spmv_task_completed)[t]->dev_m * sizeof(double))); 

	}
}

void print_task_info(spmv_task * t) {
	cout << "start_idx = " << t->start_idx << endl;
	cout << "end_idx = " << t->end_idx << endl;
}
