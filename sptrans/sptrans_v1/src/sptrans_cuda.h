#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <assert.h>

using namespace std;


int cuda_sptrans(const int         m,
                          const int         n,
                          const int         nnz,
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
// get csc col pointer, nvidia's has some problem on this value
/*
    memset(cscColPtr, 0, sizeof(int) * (n+1));

    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }
    exclusive_scan(cscColPtr, n + 1);
// 
*/
    cudaSetDevice(0);
    
    cusparseHandle_t handle = NULL;
    cudaStream_t stream = NULL;

    
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;


   // cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
   // assert(cudaSuccess == cudaStat1);

    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);



    //transfer csc to csr 
    struct timeval t10, t11;
    double time_memory_copy=0;
    gettimeofday(&t10,NULL);
    
    // transfer host mem to device mem
    int *d_csrRowPtr;
    int *d_csrColIdx;
    double *d_csrVal;
   
    int *d_cscColPtr;
    int *d_cscRowIdx;
    double *d_cscVal;

    // Matrix csr
    cudaMallocManaged((void **)&d_csrRowPtr, (m+1) * sizeof(int));
    cudaMallocManaged((void **)&d_csrColIdx, nnz  * sizeof(int));
    cudaMallocManaged((void **)&d_csrVal,    nnz  * sizeof(double));

    cudaMemcpy(d_csrRowPtr, csrRowPtr, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdx, nnz  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal,    csrVal,    nnz  * sizeof(double),   cudaMemcpyHostToDevice);


     // Matrix csc
      
    cudaMallocManaged((void **)&d_cscColPtr, (n+1) * sizeof(int));
    cudaMallocManaged((void **)&d_cscRowIdx, nnz  * sizeof(int));
    cudaMallocManaged((void **)&d_cscVal,    nnz  * sizeof(double));

    cudaMemset(d_cscColPtr, 0, (n+1) * sizeof(int));
    cudaMemset(d_cscRowIdx, 0, nnz  * sizeof(int));
    cudaMemset(d_cscVal,    0,    nnz  * sizeof(double));


    gettimeofday(&t11, NULL);
    time_memory_copy = (t11.tv_sec - t10.tv_sec) * 1000.0 + (t11.tv_usec - t10.tv_usec) / 1000.0;

    printf("cuSparse memory copy for single gpu used %4.2f ms,\n",time_memory_copy);

    struct timeval t3, t4;
    double time_cuda_trans= 0;
    gettimeofday(&t3, NULL); 
    
    // setup buffersize

    size_t P_bufferSize = 0;

    char* p_buffer= NULL;

    status = cusparseCsr2cscEx2_bufferSize(handle,
                              m,
                              n,
                              nnz,
                              d_csrVal,
                              d_csrRowPtr,
                              d_csrColIdx,
                              d_cscVal,
                              d_cscColPtr,
                              d_cscRowIdx,
                              CUDA_C_32F,
                              CUSPARSE_ACTION_NUMERIC,
                              CUSPARSE_INDEX_BASE_ZERO,
                              CUSPARSE_CSR2CSC_ALG1,
                              &P_bufferSize);

    printf("P_bufferSize  = %lld \n", (long long)P_bufferSize);
    if (NULL != p_buffer) { cudaFree(p_buffer); }
    cudaStat1 = cudaMalloc((void**)&p_buffer, P_bufferSize);
    assert(cudaSuccess == cudaStat1);
    
    status = cusparseCsr2cscEx2(
                handle,
                m,
                n,
                nnz,
	            	d_csrVal,
                d_csrRowPtr,
                d_csrColIdx,
                d_cscVal,
                d_cscColPtr,
                d_cscRowIdx,
		CUDA_C_32F,
                CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO,
		CUSPARSE_CSR2CSC_ALG1,
		p_buffer);
    //assert(CUSPARSE_STATUS_SUCCESS == status);
    
    if (CUSPARSE_STATUS_INTERNAL_ERROR == status) printf("CUSPARSE_STATUS_INTERNAL_ERROR\n");
    //assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    //assert(cudaSuccess == cudaStat1);
    //cudaDeviceSynchronize();
    gettimeofday(&t4, NULL);
    time_cuda_trans = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    
    printf("cuSparse trans used %4.2f ms,\n",time_cuda_trans);
   
    
   
    cudaMemcpy(cscColPtr, d_cscColPtr, (n+1) * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(cscRowIdx, d_cscRowIdx, nnz  * sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(cscVal, d_cscVal,   nnz  * sizeof(double),   cudaMemcpyDeviceToHost);

    // validate value

    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < nnz; i++)
    {
        ref += abs(cscVal[i]);
        res += abs(cscVal_ref[i] - cscVal[i]);
        if (cscVal_ref[i] != cscVal[i]) printf ("%i, [%d, %d] cscValA = %f, cscValB = %f\n", i, cscRowIdx_ref[i],  cscRowIdx[i], cscVal_ref[i], cscVal[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("sptrans value test on single GPU: passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrans value test on single GPU: _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);


  ref = 0.0;
  res = 0.0;
 
	for (int i = 0; i < n+1; i++)
    {
        ref += abs(cscColPtr[i]);
        res += abs(cscColPtr_ref[i] - cscColPtr[i]);
        if (cscColPtr_ref[i] != cscColPtr[i]) printf ("%i, cscColPtr_ref = %i, cscColPtr = %i\n", i, cscColPtr_ref[i], cscColPtr[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("sptrans pointer test on single GPU: pasted! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("sptrans pointer test on single GPU: _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

   

    // step 6: free resources
    
   
    cusparseDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscRowIdx);
    cudaFree(d_cscVal);
    
    

    return 0;
}



