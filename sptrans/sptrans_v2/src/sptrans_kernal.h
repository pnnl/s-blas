#include "common.h"
#include "utils.h"
//#include <bits/stdc++.h> 
#include <cuda_runtime.h>
//#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <assert.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;


__global__
void sptrans_cuda_ptr(const int   *dev_csrColIdx,
                                   const int    d_nnz,
                                         int   *cscColPtr,
					 int   *dev_cloc,
                                   const int  start_nnz)
{
    const int device_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    const int global_id = device_id + start_nnz;   
   // printf("global_id: %d\n", global_id);	

    if(device_id < d_nnz)
    {
        //__threadfence_system();
         dev_cloc[device_id] = atomicAdd_system(&cscColPtr[dev_csrColIdx[device_id]], 1);
        //__threadfence_system();
       
    }
}



__global__
void sptrans_cuda_value(  const int        *dev_csrRowPtr,
                          const int        *dev_csrColIdx,
                          const VALUE_TYPE *dev_csrVal,
				int        *cscRowIdx,
                          const int        *cscColPtr,
                                VALUE_TYPE *cscVal,
			  const int   *dev_cloc,   
                          const int    n,
                          const int    d_nnz,  
                          const int  start_nnz,
			  const int  start_row,
			  const int  d_id)
{
     const int device_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
     const int global_id = device_id + start_nnz;  

      if(device_id < d_nnz)
      {
	    int i =0;
            do{
                i++;
	    }while(device_id >= dev_csrRowPtr[i]); // get the row idx
            int row = i-1;

	    
            int loc = cscColPtr[dev_csrColIdx[device_id]]+ dev_cloc[device_id];  //get the loc

	    cscRowIdx[loc] = row+start_row;
	    cscVal[loc]=dev_csrVal[device_id];
      }
}


__global__
void sptrans_cuda_sort( 	int        *cscRowIdx,
                          const int        *cscColPtr,
                                double     *cscVal,
			  const int n,
			  const int d_id)
{
	const int global_id= threadIdx.x+ blockDim.x* (blockIdx.x+ d_id* gridDim.x);
      	if(global_id < n){
		const int begin = cscColPtr[global_id];
		const int end = cscColPtr[global_id+1];
                int col_length = end - begin;
	// do sort in key value between begin and end
		int *keys = new int [col_length];
		double *value = new double [col_length];
		
		for (int i = 0; i<col_length; i++){
			keys[i] = cscRowIdx[begin+i];
			value[i] = cscVal[begin+i];
		}

		thrust::sort_by_key(thrust::seq, keys, keys+col_length, value);
		
		for (int i = 0; i<col_length; i++){
			cscRowIdx[begin+i] = keys[i];
			cscVal[begin+i] = value[i];
		}
	}
}


int kernal_sptrans(const int         m,
                          const int         n,
                          const int         nnz,
                          int              ngpu,
                          const int        *csrRowPtr,
                          const int        *csrColIdx,
                          const VALUE_TYPE *csrVal,
                                int        *cscRowIdx,
                                int        *cscColPtr,
                                VALUE_TYPE *cscVal,
			  const int        *cscRowIdx_ref,
                          const int        *cscColPtr_ref,
			  const VALUE_TYPE *cscVal_ref)
{

   
    int num_threads = 128;
    int num_blocks = ceil ((double)nnz / (double)num_threads) ; 

    cudaStream_t * stream = new cudaStream_t [ngpu];

	cudaError_t * cudaStat1 = new cudaError_t[ngpu];
	cudaError_t * cudaStat2 = new cudaError_t[ngpu];
	cudaError_t * cudaStat3 = new cudaError_t[ngpu];
	cudaError_t * cudaStat4 = new cudaError_t[ngpu];
	cudaError_t * cudaStat5 = new cudaError_t[ngpu];
	cudaError_t * cudaStat6 = new cudaError_t[ngpu];
	

    //initial
    struct timeval t3, t4;
    double time_cuda_setup= 0;
    gettimeofday(&t3, NULL);

        int  * start_row  = new int[ngpu];
	int  * end_row    = new int[ngpu];
	int  * start_nnz  = new int[ngpu];
	
	int * dev_m            = new int      [ngpu];
	int * dev_n            = new int      [ngpu];
	int * dev_nnz          = new int      [ngpu];

	int ** host_csrRowPtr  = new int    * [ngpu];
	int ** dev_csrRowPtr   = new int    * [ngpu];
	int ** dev_csrColIdx   = new int    * [ngpu];
	double ** dev_csrVal   = new double * [ngpu];

	int **dev_cloc         = new int    * [ngpu];

	int * dev_cscColPtr;                                 //put the csc matrix in unified memory
	int * dev_cscRowIdx;
	double * dev_cscVal;


//distribute csr to csr[d]

        for (int d = 0; d < ngpu; d++){     

		//cudaSetDevice(d);

		start_row[d] = floor((d)     * m / ngpu);
		end_row[d]   = floor((d + 1) * m / ngpu) - 1;

		dev_m[d]   = end_row[d] - start_row[d] + 1;
		dev_n[d]   = n;
		dev_nnz[d] = (int)(csrRowPtr[end_row[d] + 1] - csrRowPtr[start_row[d]]);

 		if (d == 0){
    		  	start_nnz[d] = 0;   
    		}
   		else{
    		  	start_nnz[d] = start_nnz[d-1] + dev_nnz[d-1];
    		}
		host_csrRowPtr[d] = new int[dev_m[d] + 1];
		for (int i = 0; i < dev_m[d] + 1; i++) {
			host_csrRowPtr[d][i] = (int)(csrRowPtr[start_row[d] + i] - csrRowPtr[start_row[d]]);
		}

	}

	gettimeofday(&t4, NULL);
	time_cuda_setup = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    
 	printf("cuSparse setup matrix used %4.8f ms,\n",time_cuda_setup);
    
// copy csr matrix to device


        struct timeval t5, t6;
        double time_cuda_mem= 0;
        gettimeofday(&t5, NULL);
    
        //allocate csc matrix to unified memory

	cudaStat1[0] = cudaMallocManaged((void**)&dev_cscColPtr,   (n + 1) * sizeof(int));
	cudaStat2[0] = cudaMallocManaged((void**)&dev_cscRowIdx,   nnz * sizeof(int)); 
	cudaStat3[0] = cudaMallocManaged((void**)&dev_cscVal,      nnz * sizeof(double)); 


	cudaStat4[0] = cudaMemset(dev_cscColPtr,  0,  (n + 1) * sizeof(int));
	cudaStat5[0] = cudaMemset(dev_cscRowIdx,  0,  nnz * sizeof(int)); 
	cudaStat6[0] = cudaMemset(dev_cscVal,     0,  nnz * sizeof(double));
	
	if ((cudaStat1[0] != cudaSuccess) || 
			(cudaStat3[0] != cudaSuccess) ||
			(cudaStat4[0] != cudaSuccess) ||
			(cudaStat5[0] != cudaSuccess) ||
			(cudaStat6[0] != cudaSuccess) ||
			(cudaStat2[0] != cudaSuccess))
			
  
		{ 
			printf("Device malloc for csc failed");
			return 1; 
		} 

	// allocate csr value and pointer
        for (int d = 0; d < ngpu; d++){
		cudaSetDevice(d);

		cudaStat1[d] = cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int));
		cudaStat2[d] = cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d] * sizeof(double)); 
		cudaStat3[d] = cudaMalloc((void**)&dev_csrColIdx[d], dev_nnz[d] * sizeof(int)); 
		cudaStat4[d] = cudaMalloc((void**)&dev_cloc[d], dev_nnz[d] * sizeof(int)); 

		

		if ((cudaStat1[d] != cudaSuccess) || 
			(cudaStat3[d] != cudaSuccess) ||
			(cudaStat4[d] != cudaSuccess) ||
			(cudaStat2[d] != cudaSuccess))
			
  
		{ 
			printf("Device malloc failed");
			return 1; 
		} 

		//cout << "Start copy to GPUs...";
		cudaStat1[d] = cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],                  (dev_m[d] + 1) * sizeof(int), cudaMemcpyHostToDevice); 
		cudaStat2[d] = cudaMemcpy(dev_csrVal[d],      &csrVal[csrRowPtr[start_row[d]]],    dev_nnz[d] * sizeof(double), cudaMemcpyHostToDevice);
		cudaStat3[d] = cudaMemcpy(dev_csrColIdx[d],   &csrColIdx[csrRowPtr[start_row[d]]], dev_nnz[d] * sizeof(int),   cudaMemcpyHostToDevice);
		cudaStat4[d] = cudaMemset(dev_cloc[d],  0,  dev_nnz[d] * sizeof(int)); 
		
		
		

		if ((cudaStat1[d] != cudaSuccess) ||
			(cudaStat3[d] != cudaSuccess) ||
			(cudaStat4[d] != cudaSuccess) ||
		 	(cudaStat2[d] != cudaSuccess))
		{ 
			printf("Memcpy from Host to Device failed"); 
			return 1; 
		} 

	}

	gettimeofday(&t6, NULL);
	time_cuda_mem = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
    
 	printf("cuSparse copy matrix used %4.8f ms,\n",time_cuda_mem);


   
// sparse calculate cscColPtr on multiple GPU
//void sptrans_cuda_ptr(const int   *dev_csrColIdx,
//                                   const int    d_nnz,
//                                         int   *cscColPtr,
//					 int   *dev_cloc,
//                                   const int  start_nnz)

        struct timeval t7, t8;
        double time_cuda_trans= 0;
        gettimeofday(&t7, NULL);


	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		num_blocks = ceil ((double)dev_nnz[d] / (double)num_threads);
	      	sptrans_cuda_ptr<<< num_blocks, num_threads >>>
                                      (dev_csrColIdx[d], dev_nnz[d], dev_cscColPtr, dev_cloc[d], start_nnz[d]);	 	 	
	}

	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
	}
    
   	gettimeofday(&t8, NULL);
	time_cuda_trans = (t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;
    
 	printf("cuSparse point generate on multiple gpu used %4.8f ms,\n",time_cuda_trans);

        exclusive_scan(dev_cscColPtr, n + 1);

    
//  sparse calculate cscVal on multiple GPU
/*
__global__
void sptrans_cuda_value(  const int        *dev_csrRowPtr,
                          const int        *dev_csrColIdx,
                          const VALUE_TYPE *dev_csrVal,
				int        *cscRowIdx,
                          const int        *cscColPtr,
                                VALUE_TYPE *cscVal,
			  const int   *dev_cloc,   
                          const int    n,
                          const int    d_nnz,  
                          const int  start_nnz,
			  const int  d_id)
{
*/
	 struct timeval t9, t10;
        double time_cuda_comps= 0;
        gettimeofday(&t9, NULL);


 
	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		num_blocks = ceil ((double)dev_nnz[d] / (double)num_threads);
	      	sptrans_cuda_value<<< num_blocks, num_threads >>>
                                      (dev_csrRowPtr[d],dev_csrColIdx[d], dev_csrVal[d], dev_cscRowIdx, dev_cscColPtr, dev_cscVal, dev_cloc[d], dev_n[d], dev_nnz[d],  start_nnz[d], start_row[d], d);	 	 	
	}

	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
	}

	gettimeofday(&t10, NULL);
	time_cuda_comps = (t10.tv_sec - t9.tv_sec) * 1000.0 + (t10.tv_usec - t9.tv_usec) / 1000.0;
    
 	printf("cuSparse value insert used %4.8f ms,\n", time_cuda_comps);
  



// sparse sort in each colum
/*__global__
void sptrans_cuda_sort( 	int        *cscRowIdx,
                          const int        *cscColPtr,
                                VALUE_TYPE *cscVal
			)
*/



       struct timeval t11, t12;
        double time_cuda_sort= 0;
        gettimeofday(&t11, NULL);


 
	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		num_blocks = ceil ((double) n / (double)(num_threads*ngpu));
	      	sptrans_cuda_sort<<< num_blocks, num_threads >>>
                                      (dev_cscRowIdx, dev_cscColPtr, dev_cscVal,n,d);	 	 	
	}

	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
	}

/*
//host sort

        cudaSetDevice(0);

	 double *dev_cscVal_sorted;
	cudaMallocManaged((void **)&dev_cscVal_sorted,    nnz  * sizeof(double));
	cudaMemset(dev_cscVal_sorted,    0,    nnz  * sizeof(double));

	size_t pBufferSizeInBytes = 0;
	void *pBuffer = NULL;
	int *P = NULL;
	cusparseHandle_t handle = NULL;
	
	
	cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
       


        cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking);
   
    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

     cudaStat1[0] = cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking);
     assert(cudaSuccess == cudaStat1[0]);


    status = cusparseSetStream(handle, stream[0]);
    assert(CUSPARSE_STATUS_SUCCESS == status);


	cusparseMatDescr_t descrA = NULL;

	cusparseCreateMatDescr(&descrA);

	//cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	
   
	// step 1: allocate buffer
	cusparseXcscsort_bufferSizeExt(handle, m, n, nnz, dev_cscColPtr, dev_cscRowIdx, &pBufferSizeInBytes);
	cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);

	// step 2: setup permutation vector P to identity
	cudaMalloc( (void**)&P, sizeof(int)*nnz);
	cusparseCreateIdentityPermutation(handle, nnz, P);

	// step 3: sort CSC format
	cusparseXcscsort(handle, m, n, nnz, descrA, dev_cscColPtr, dev_cscRowIdx, P, pBuffer);

	// step 4: gather sorted cscVal
	cusparseDgthr(handle, nnz, dev_cscVal, dev_cscVal_sorted, P, CUSPARSE_INDEX_BASE_ZERO);
*/
	gettimeofday(&t12, NULL);
	time_cuda_sort = (t12.tv_sec - t11.tv_sec) * 1000.0 + (t12.tv_usec - t11.tv_usec) / 1000.0;
    
 	printf("cuSparse sort in index used %4.8f ms,\n", time_cuda_sort);

    printf("SpTrans computation time: %.3f ms \n", time_cuda_sort + time_cuda_trans + time_cuda_comps);


     //copy value
   /*
     cudaStat1[0] = cudaMemcpy(&cscVal, dev_cscVal, nnz * sizeof(double), cudaMemcpyDeviceToHost);
     cudaStat2[0] = cudaMemcpy(&cscRowIdx, dev_cscRowIdx, nnz * sizeof(int), cudaMemcpyDeviceToHost);	
     cudaStat3[0] = cudaMemcpy(&cscColPtr, dev_cscColPtr, (n+1)* sizeof(int), cudaMemcpyDeviceToHost);	
	
     
	if ((cudaStat1[0] != cudaSuccess) || 
			(cudaStat3[0] != cudaSuccess) ||
			(cudaStat2[0] != cudaSuccess))
			
  
		{ 
			printf("Device copy back to host for csc failed");
			return 1; 
		} 
    */

	for (int i = 0; i < nnz; i++){
	cscVal[i] = dev_cscVal[i];
	cscRowIdx[i] = dev_cscRowIdx[i];
	}

	for (int i = 0; i < (n+1); i++){
	cscColPtr[i] = dev_cscColPtr[i];
	}

     // validate value


    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < nnz; i++)
    {
        ref += abs(cscVal[i]);
        res += abs(cscVal_ref[i] - cscVal[i]);
 //       if (cscVal_ref[i] != cscVal[i]) printf ("%i, [%d, %d] cscValA = %f, cscValB = %f\n", i, cscRowIdx_ref[i],  cscRowIdx[i], cscVal_ref[i], cscVal[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("sptrans value test on multiple GPU: passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrans value test on multiple GPU: _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);


    ref = 0.0;
    res = 0.0;

	for (int i = 0; i < n+1; i++)
    {
        ref += abs(cscColPtr[i]);
        res += abs(cscColPtr_ref[i] - cscColPtr[i]);
//        if (cscColPtr_ref[i] != cscColPtr[i]) printf ("%i, cscColPtr_ref = %i, cscColPtr = %i\n", i, cscColPtr_ref[i], cscColPtr[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("sptrans pointer test on multiple GPU: passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrans pointer test on multiple GPU: _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);


    // free resources
    
   
	for (int d = 0; d < ngpu; d++) {
		cudaSetDevice(d);
		cudaFree(dev_csrVal[d]);
		cudaFree(dev_csrRowPtr[d]);
		cudaFree(dev_csrColIdx[d]);
		cudaFree(dev_cloc[d]);
		
	}

	cudaFree(dev_cscVal);
	cudaFree(dev_cscColPtr);
	cudaFree(dev_cscRowIdx);

    

    return 0;
}



