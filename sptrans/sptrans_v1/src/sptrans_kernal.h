#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
//#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <assert.h>

using namespace std;



__global__
void sptrans_cuda_composition_ptr(const int  *cscColPtr_temp,
                           	 int   *m_cscColPtr,
				 const int ngpu,
		      		 const int   n)
{
    int device_id = blockIdx.x * blockDim.x + threadIdx.x; //get_thread_id	


    if(device_id < n)
    {
        	for(int d = 0; d < ngpu; d++){
		int pos = d*(n+1)+device_id+1;
		 m_cscColPtr[device_id+1] += cscColPtr_temp[pos];    //calculated cscColPtr
		 //printf("dev id %d, pointer %d : %d\n", device_id, m_cscColPtr[device_id+1], cscColPtr_temp[pos]);
    	         }   
    }
     
   // printf("pointer is %d\n",m_cscColPtr[device_id+1]);

}

//sptrans_cuda_composition_value<<< num_blocks, num_threads >>>
         //                            (cscColPtr_temp, cscRowIdx_temp, cscVal_temp, m_cscColPtr, m_cscRowIdx, m_cscVal, dev_nnz, ngpu, n);

__global__
void sptrans_cuda_composition_value(const int        *cscColPtr,
                          const int        *cscRowIdx,
                          const VALUE_TYPE *cscVal,
                          const int        *m_cscColPtr,
				int	   *m_cscRowIdx,
                                VALUE_TYPE *m_cscVal,
			  const int *start_nnz,
			  const int ngpu,
			  const int n)
{
     int device_id = blockIdx.x * blockDim.x + threadIdx.x; //get_thread_id;
    // const int global_id = device_id + start_nnz;  

      if(device_id < n)
      {
	         int start = m_cscColPtr[device_id];   //nnz for each col
		 int k = 0;
                 

		for(int j = 0; j<ngpu; j++){
		//int j =0;
		      int pos_ptr = j*(n+1);
		      int pos = start_nnz[j];
                      int cscColinc = cscColPtr[pos_ptr+device_id+1] - cscColPtr[pos_ptr+device_id];
		      int dev_start = cscColPtr[pos_ptr+device_id];
                      for(int i =0; i < cscColinc; i++){
			m_cscRowIdx[start+k] = cscRowIdx[(pos+dev_start+i)];
			m_cscVal[start+k] = cscVal[(pos+dev_start+i)];
			k++;
			} 
			//pos = pos + dev_nnz[j];
		
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

   

    
    cudaStream_t * stream = new cudaStream_t [ngpu];

	cudaError_t * cudaStat1 = new cudaError_t[ngpu];
	cudaError_t * cudaStat2 = new cudaError_t[ngpu];
	cudaError_t * cudaStat3 = new cudaError_t[ngpu];
	cudaError_t * cudaStat4 = new cudaError_t[ngpu];
	cudaError_t * cudaStat5 = new cudaError_t[ngpu];
	cudaError_t * cudaStat6 = new cudaError_t[ngpu];

	cusparseStatus_t * status = new cusparseStatus_t[ngpu];
	cusparseHandle_t * handle = new cusparseHandle_t[ngpu];



    //transfer csc to csr 
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
	int ** dev_csrColIdx = new int    * [ngpu];
	double ** dev_csrVal   = new double * [ngpu];

	int ** dev_cscColPtr   = new int    * [ngpu];
	int ** dev_cscRowIdx  = new int    * [ngpu];
	double ** dev_cscVal   = new double * [ngpu];


	double ** dev_x = new double * [ngpu];
	double ** dev_y = new double * [ngpu];

//distribute csr to csr[d]

        for (int d = 0; d < ngpu; d++){     

		cudaSetDevice(d);

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
		

		cudaStat1[d] = cudaMalloc((void**)&dev_csrRowPtr[d],   (dev_m[d] + 1) * sizeof(int));
		cudaStat2[d] = cudaMalloc((void**)&dev_csrColIdx[d], dev_nnz[d] * sizeof(int)); 
		cudaStat3[d] = cudaMalloc((void**)&dev_csrVal[d],      dev_nnz[d] * sizeof(double)); 
		
		cudaStat4[d] = cudaMalloc((void**)&dev_cscColPtr[d],   (dev_n[d] + 1) * sizeof(int));
		cudaStat5[d] = cudaMalloc((void**)&dev_cscRowIdx[d],   dev_nnz[d] * sizeof(int)); 
		cudaStat6[d] = cudaMalloc((void**)&dev_cscVal[d],      dev_nnz[d] * sizeof(double)); 

		
		

		if ((cudaStat1[d] != cudaSuccess) || 
			(cudaStat2[d] != cudaSuccess) || 
			(cudaStat3[d] != cudaSuccess) || 
			(cudaStat4[d] != cudaSuccess) || 
			(cudaStat5[d] != cudaSuccess) ||
			(cudaStat6[d] != cudaSuccess))
  
		{ 
			printf("Device malloc failed");
			return 1; 
		} 

		//cout << "Start copy to GPUs...";
		cudaStat1[d] = cudaMemcpy(dev_csrRowPtr[d],   host_csrRowPtr[d],                  (dev_m[d] + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaStat2[d] = cudaMemcpy(dev_csrColIdx[d],   &csrColIdx[csrRowPtr[start_row[d]]], dev_nnz[d] * sizeof(int),   cudaMemcpyHostToDevice); 
		cudaStat3[d] = cudaMemcpy(dev_csrVal[d],      &csrVal[csrRowPtr[start_row[d]]],    dev_nnz[d] * sizeof(double), cudaMemcpyHostToDevice);

		cudaStat4[d] = cudaMemset(dev_cscColPtr[d],  0,  (dev_n[d] + 1) * sizeof(int));
		cudaStat5[d] = cudaMemset(dev_cscRowIdx[d],  0,  dev_nnz[d] * sizeof(int)); 
		cudaStat6[d] = cudaMemset(dev_cscVal[d],     0,  dev_nnz[d] * sizeof(double));
		
		

		if ((cudaStat1[d] != cudaSuccess) ||
		 	(cudaStat2[d] != cudaSuccess) ||
		  	(cudaStat3[d] != cudaSuccess) ||
		   	(cudaStat4[d] != cudaSuccess) ||
		    	(cudaStat5[d] != cudaSuccess) ||
			(cudaStat6[d] != cudaSuccess))
		{ 
			printf("Memcpy from Host to Device failed"); 
			return 1; 
		} 

	}

	gettimeofday(&t6, NULL);
	time_cuda_mem = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
    
 	printf("cuSparse copy matrix used %4.8f ms,\n",time_cuda_mem);


   
// sparse transposition on multiple gpu


        struct timeval t7, t8;
        double time_cuda_trans= 0;
        gettimeofday(&t7, NULL);


	 size_t* P_bufferSize = new size_t [ngpu];

  	 char** p_buffer= new char * [ngpu];

        for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		status[d] = cusparseCsr2cscEx2_bufferSize(
			handle[d],
                	dev_m[d],
                	dev_n[d],
              	  	dev_nnz[d],
			dev_csrVal[d],
		        dev_csrRowPtr[d],
                	dev_csrColIdx[d],
                	dev_cscVal[d],
                	dev_cscColPtr[d],
                	dev_cscRowIdx[d],
		 	CUDA_C_32F,
                	CUSPARSE_ACTION_NUMERIC,
                	CUSPARSE_INDEX_BASE_ZERO,
			CUSPARSE_CSR2CSC_ALG1,
                        &P_bufferSize[d]);

                              

    			printf("P_bufferSize  in gpu %d = %lld \n", d, (long long)P_bufferSize[d]);
    			if (NULL != p_buffer[d]) { cudaFree(p_buffer[d]); }
   			 cudaStat1[d] = cudaMalloc((void**)&p_buffer[d], P_bufferSize[d]);
    			assert(cudaSuccess == cudaStat1[d]);
/*
	}
	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
//		assert(CUSPARSE_STATUS_SUCCESS == status[d]);
    
    		if (CUSPARSE_STATUS_INTERNAL_ERROR == status[d]) printf("CUSPARSE_STATUS_INTERNAL_ERROR\n");
  //  		assert(CUSPARSE_STATUS_SUCCESS == status[d]);
	}


	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);	
*/
		status[d] = cusparseCsr2cscEx2(
                handle[d],
                dev_m[d],
                dev_n[d],
                dev_nnz[d],
		dev_csrVal[d],
                dev_csrRowPtr[d],
                dev_csrColIdx[d],
                dev_cscVal[d],
                dev_cscColPtr[d],
                dev_cscRowIdx[d],
		CUDA_C_32F,
                CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO,
		CUSPARSE_CSR2CSC_ALG1,
		p_buffer[d]);	 	 	
	}

	for (int d = 0; d < ngpu; ++d) 
	{
		cudaSetDevice(d);
		cudaDeviceSynchronize();
//		assert(CUSPARSE_STATUS_SUCCESS == status[d]);
    
    		if (CUSPARSE_STATUS_INTERNAL_ERROR == status[d]) printf("CUSPARSE_STATUS_INTERNAL_ERROR\n");
  //  		assert(CUSPARSE_STATUS_SUCCESS == status[d]);
	}
    
   	gettimeofday(&t8, NULL);
	time_cuda_trans = (t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;
    
 	printf("cuSparse transposition on multiple gpu used %4.8f ms,\n",time_cuda_trans);

    
//  copy back to host and composition

   

	 struct timeval t9, t10;
        double time_cuda_comps= 0;
        gettimeofday(&t9, NULL);



// CPU composition
     
/*
    int **cscColPtr_temp = new int    * [ngpu];	
    memset(cscColPtr, 0, sizeof(int) * (n+1));

    for (int d = 0; d < ngpu; d++)
	{


	
		cscColPtr_temp[d] = (int *)malloc((n+1)* sizeof(int));          // calcolate this in cpu
		memset (cscColPtr_temp[d], 0, sizeof(int) * (n+1));		
	  cudaMemcpy(cscColPtr_temp[d], dev_cscColPtr[d], (n+1)*sizeof(int),  cudaMemcpyDeviceToHost);

		for (int i = 0; i < n+1; i++){
		cscColPtr[i] += cscColPtr_temp[d][i];    //calculated cscColPtr
                //nnz_temp += cscColPtr_temp[d][i];
                }

    }
	
   //2.2 copy index and value

     int k = 0;   //nnz idx
     int *dev_k = new int [ngpu];
     
     for (int d = 0; d < ngpu; d++){
    	dev_k[d] = 0;
  	}

     for (int col = 0; col < n; col++)
     {
	    	for (int d = 0; d < ngpu; d++)
	    	{
             int cscColinc = cscColPtr_temp[d][col+1] - cscColPtr_temp[d][col];
             
		  	      cudaMemcpy(&cscRowIdx[k], &dev_cscRowIdx[d][dev_k[d]], cscColinc*sizeof(int), cudaMemcpyDeviceToHost);
              cudaMemcpy(&cscVal[k], &dev_cscVal[d][dev_k[d]], cscColinc*sizeof(double), cudaMemcpyDeviceToHost);
		        	k += cscColinc;
		        	dev_k[d] += cscColinc;
		  	     
	    	}
     }
*/

// GPU composition
    cudaSetDevice(0);

    int *cscColPtr_temp;   //define csc_temp
    int *cscRowIdx_temp;
    double *cscVal_temp;
    int *dev_start_nnz;

    cudaStat1[0] = cudaMalloc((void**)&cscColPtr_temp,   ngpu * (n + 1) * sizeof(int));
    cudaStat2[0] = cudaMalloc((void**)&cscRowIdx_temp,   nnz * sizeof(int)); 
    cudaStat3[0] = cudaMalloc((void**)&cscVal_temp,      nnz * sizeof(double)); 
    cudaStat4[0] = cudaMalloc((void**)&dev_start_nnz,   ngpu * sizeof(int)); 


    

	for (int d = 0; d < ngpu; d++) // copy all to master GPU
	{
    		cudaStat1[d] = cudaMemcpy(&cscRowIdx_temp[start_nnz[d]], dev_cscRowIdx[d], dev_nnz[d]*sizeof(int),  cudaMemcpyDeviceToDevice);
		cudaStat2[d] = cudaMemcpy(&cscVal_temp[start_nnz[d]], dev_cscVal[d], dev_nnz[d]*sizeof(double),  cudaMemcpyDeviceToDevice);    		
		cudaStat3[d] = cudaMemcpy(&cscColPtr_temp[d*(n+1)], dev_cscColPtr[d], (n+1)*sizeof(int),  cudaMemcpyDeviceToDevice);   

	}
	
	cudaMemcpy(dev_start_nnz, start_nnz, ngpu * sizeof(int), cudaMemcpyHostToDevice);


    int *m_cscColPtr;   //define master csc
    int *m_cscRowIdx;
    double *m_cscVal;

    	cudaStat1[0] = cudaMalloc((void**)&m_cscColPtr,   (n + 1) * sizeof(int));
	cudaStat2[0] = cudaMalloc((void**)&m_cscRowIdx,   nnz * sizeof(int)); 
	cudaStat3[0] = cudaMalloc((void**)&m_cscVal,      nnz * sizeof(double)); 


	cudaStat4[0] = cudaMemset(m_cscColPtr,  0,  (n + 1) * sizeof(int));
	cudaStat5[0] = cudaMemset(m_cscRowIdx,  0,  nnz * sizeof(int)); 
	cudaStat6[0] = cudaMemset(m_cscVal,     0,  nnz * sizeof(double));
	
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

	
        int num_threads = 128; 	
	int num_blocks = ceil ((double) (n) / (double)num_threads);
	sptrans_cuda_composition_ptr<<< num_blocks, num_threads >>>
                                      (cscColPtr_temp, m_cscColPtr,ngpu, n);	 	 	
	

    
	//cudaDeviceSynchronize();

	//num_blocks = ceil ((double) nnz / (double)num_threads);
	sptrans_cuda_composition_value<<< num_blocks, num_threads >>>
                                    (cscColPtr_temp, cscRowIdx_temp, cscVal_temp, m_cscColPtr, m_cscRowIdx, m_cscVal, dev_start_nnz, ngpu, n);


	//cudaDeviceSynchronize();
    
// copy back

    cudaMemcpy(cscColPtr, m_cscColPtr, (n+1) * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, m_cscRowIdx, nnz  * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal, m_cscVal,   nnz  * sizeof(double),   cudaMemcpyDeviceToHost);


    gettimeofday(&t10, NULL);
	  time_cuda_comps = (t10.tv_sec - t9.tv_sec) * 1000.0 + (t10.tv_usec - t9.tv_usec) / 1000.0;
    
 	  printf("cuSparse composition used %4.8f ms,\n", time_cuda_comps);


 	printf("SpTrans computation time: %.3f ms \n",time_cuda_trans + time_cuda_comps);

  
     // validate value

    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < nnz; i++)
    {
        ref += abs(cscVal[i]);
        res += abs(cscVal_ref[i] - cscVal[i]);
        //if (cscVal_ref[i] != cscVal[i]) printf ("%i, [%d, %d] cscValA = %f, cscValB = %f\n", i, cscRowIdx_ref[i],  cscRowIdx[i], cscVal_ref[i], cscVal[i]);
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
        //if (cscColPtr_ref[i] != cscColPtr[i]) printf ("%i, cscColPtr_ref = %i, cscColPtr = %i\n", i, cscColPtr_ref[i], cscColPtr[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("sptrans pointer test on multiple GPU: passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrans pointer test on multiple GPU: _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);


    // free resources
    
   
	for (int d = 0; d < ngpu; d++) {
		cudaSetDevice(d);
		cusparseDestroy(handle[d]);
    		cudaStreamDestroy(stream[d]);
		cudaFree(dev_csrVal[d]);
		cudaFree(dev_csrRowPtr[d]);
		cudaFree(dev_csrColIdx[d]);
		cudaFree(dev_cscVal[d]);
		cudaFree(dev_cscColPtr[d]);
		cudaFree(dev_cscRowIdx[d]);
	}
   
	cudaSetDevice(0);
  cudaFree(m_cscVal);
	cudaFree(m_cscColPtr);
	cudaFree(m_cscRowIdx);
	cudaFree(cscVal_temp);
	cudaFree(cscColPtr_temp);
	cudaFree(cscRowIdx_temp);

	delete[] dev_csrVal;
	delete[] dev_csrRowPtr;
	delete[] dev_csrColIdx;
	delete[] dev_cscVal;
	delete[] dev_cscColPtr;
	delete[] dev_cscRowIdx;
	delete[] host_csrRowPtr;
	delete[] start_row;
	delete[] end_row;

    
    

    return 0;
}



