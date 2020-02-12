#ifndef _SPTRSV_SYNCFREE_CUDA_
#define _SPTRSV_SYNCFREE_CUDA_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>



__global__
void sptrsv_syncfree_cuda_analyser(const int   *cscRowIdx,
                                   const int    m,
                                   const int    nnz,
                                         int   *graphInDegree,
                                   const int  start_nnz,
					                         const int	d_id)
{
    const int device_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    const int global_id = device_id + start_nnz;   
   // printf("global_id: %d\n", global_id);	

    if (device_id < nnz)
    {
        //__threadfence_system();
        atomicAdd_system(&graphInDegree[cscRowIdx[global_id]], 1);
       //  __threadfence_system();
    }
}

__global__
void sptrsv_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                   const int* __restrict__        cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int*                     d_graphInDegree,
                                         VALUE_TYPE*              left_sum,
                                         int*                     dev_in_degree,
                                         VALUE_TYPE*              dev_left_sum,
                                   const int                      m,
                                   const int                      substitution,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     d_while_profiler,
                                   const int                      start_nnz,
                                   const int                      start_x,
				                           const int			  d_id)
					
{

   // printf("device id %d start nnz %d\n",d_id, start_nnz);
    const int device_id = blockIdx.x * blockDim.x + threadIdx.x;  //get x id
    int device_x_id =  device_id / WARP_SIZE;
    if (device_x_id >= m) {
      //  printf("m = %d, global id %d, global x id %d for device %d, block %d\n", m, global_id, global_x_id, d_id , blockIdx.x);
        return;
    }
    
    device_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  device_x_id : m - 1 - device_x_id;
                  
    //printf("global id %d, global x id %d for device %d, block %d\n", global_id, global_x_id, d_id , blockIdx.x);
    //if (global_x_id >= m) {
      //  printf("m = %d, global id %d, global x id %d for device %d, block %d\n", m, global_id, global_x_id, d_id , blockIdx.x);
    //    return;
    //}
    //const int device_id = blockIdx.x * blockDim.x + threadIdx.x ;// get device_x_id
     // substitution is forward or backward

     int starting_x = start_x;
    //int starting_x = (global_id / (WARP_PER_BLOCK * WARP_SIZE)) * WARP_PER_BLOCK; //get the first x in the device
      starting_x = substitution == SUBSTITUTION_FORWARD ? 
                  starting_x : m - 1 - starting_x;
                  
    int global_x_id = device_x_id + starting_x;
   
    
    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    // Initialize

    const int local_warp_id = threadIdx.x / WARP_SIZE;  // block x id
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
   
    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                    d_cscColPtr[device_x_id] : d_cscColPtr[device_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos]; // get x id in device to get d_cscval
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));


    if( global_x_id == 65){              
    //    printf("global x id %d device_x_id %d for device %d, block %d\n", global_x_id, device_x_id, d_id , blockIdx.x);
     //   printf("d_graphInDegree %d, local_warp_id %d starting_x %d d_cscColPtr %d d_cscVal %f \n", d_graphInDegree[global_x_id], local_warp_id, starting_x, d_cscColPtr[device_x_id],d_cscVal[pos]);
    }
    
    if (threadIdx.x < WARP_PER_BLOCK) { s_graphInDegree[threadIdx.x] = 0; s_left_sum[threadIdx.x] = 0; }
    __syncthreads;

    clock_t start = clock();
    // Consumer
    do {
        clock_t stop = clock();
        if (stop < start) break;
    }
    //while (d_graphInDegree[global_x_id] != 1);
    while ((s_graphInDegree[local_warp_id] + dev_in_degree[device_x_id] + 1) != d_graphInDegree[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory"); 
    //}
    //while (s_graphInDegree[local_warp_id] != graphInDegree );:
    
    VALUE_TYPE xi = left_sum[global_x_id] + s_left_sum[local_warp_id] + dev_left_sum[device_x_id];
    xi = (d_b[device_x_id] - xi) * coef;

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[device_x_id] : d_cscColPtr[device_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[device_x_id+1] : d_cscColPtr[device_x_id+1]-1;

    
    for (int jj = start_ptr+lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);

        const int nnzID = j + start_nnz;
        const int rowIdx = cscRowIdx[nnzID];
        
        //  printf("processing %d nnz %d lane for x %d\n", nnzID, rowIdx, global_x_id);
         //  printf("processing %d nnz %d lane for x %d\n", nnzID, rowIdx, global_x_id);
        const bool cond_inBlock = substitution == SUBSTITUTION_FORWARD ? 
                    (rowIdx < starting_x + WARP_PER_BLOCK) : (rowIdx > starting_x - WARP_PER_BLOCK);   
        const bool cond_inDevice = substitution == SUBSTITUTION_FORWARD?
                    (rowIdx < starting_x + m) : (rowIdx > starting_x - m); 
                         
        if (cond_inBlock) {
            const int pos = substitution == SUBSTITUTION_FORWARD ? 
                            rowIdx - starting_x : starting_x - rowIdx;
            atomicAdd((VALUE_TYPE *)&s_left_sum[pos], xi * d_cscVal[j]);
            __threadfence_block();
            atomicAdd((int *)&s_graphInDegree[pos], 1);
        }
        else if (cond_inDevice){
             const int pos = substitution == SUBSTITUTION_FORWARD ? 
                            rowIdx - starting_x : starting_x - rowIdx;
            atomicAdd((VALUE_TYPE *)&dev_left_sum[pos], xi * d_cscVal[j]);
            __threadfence();
            atomicAdd((int *)&dev_in_degree[pos], 1);
        }
        else {
       //  if (rowIdx == 65 ){
       //       printf("processing %d nnz %d lane for x %d InDegree is %d\n", nnzID, rowIdx, global_x_id, d_graphInDegree[rowIdx]);
       //  }     
            // __threadfence_system();  
            atomicAdd_system((VALUE_TYPE *)&left_sum[rowIdx], xi * d_cscVal[j]);
            __threadfence_system();
            atomicSub_system((int *)&d_graphInDegree[rowIdx], 1);
            // __threadfence_system();
        }
    }

    //finish
    if (!lane_id) d_x[device_x_id] = xi;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void sptrsm_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                   const int* __restrict__        d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int*                     d_graphInDegree,
                                         VALUE_TYPE*              d_left_sum,
                                   const int                      m,
                                   const int                      substitution,
                                   const int                      rhs,
                                   const int                      opt,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     d_while_profiler)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  global_x_id : m - 1 - global_x_id;

    // Initialize
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));

    clock_t start;
    // Consumer
    do {
        start = clock();
    }
    while (1 != d_graphInDegree[global_x_id]);
  
    //// Consumer
    //int graphInDegree;
    //do {
    //    //bypass Tex cache and avoid other mem optimization by nvcc/ptxas
    //    asm("ld.global.u32 %0, [%1];" : "=r"(graphInDegree),"=r"(d_graphInDegree[global_x_id]) :: "memory"); 
    //}
    //while (1 != graphInDegree );

    for (int k = lane_id; k < rhs; k += WARP_SIZE)
    {
        const int pos = global_x_id * rhs + k;
        d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
    }

    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;

    if (opt == OPT_WARP_NNZ)
    {
        for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = 0; k < rhs; k++)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_RHS)
    {
        for (int jj = start_ptr; jj < stop_ptr; jj++)
        {
            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
            const int rowIdx = d_cscRowIdx[j];
            for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
            __threadfence();
            if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
        }
    }
    else if (opt == OPT_WARP_AUTO)
    {
        const int len = stop_ptr - start_ptr;

        if ((len <= rhs || rhs > 16) && len < 2048)
        {
            for (int jj = start_ptr; jj < stop_ptr; jj++)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                if (!lane_id) atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
        else
        {
            for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
            {
                const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
                const int rowIdx = d_cscRowIdx[j];
                for (int k = 0; k < rhs; k++)
                    atomicAdd(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                __threadfence();
                atomicSub(&d_graphInDegree[rowIdx], 1);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



int sptrsv_syncfree_cuda(const int           *cscColPtrTR,
                         const int           *cscRowIdxTR,
                         const VALUE_TYPE    *cscValTR,
                         const int            m,
                         const int            n,
                         const int            nnzTR,
                         const int            substitution,
                         const int            rhs,
                         const int            opt,
                               VALUE_TYPE    *x,
                         const VALUE_TYPE    *b,
                         const VALUE_TYPE    *x_ref,
                               double        *gflops,
			       int     	     ngpu){


    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    cudaError_t *err = new cudaError_t [ngpu];
    cudaStream_t * stream = new cudaStream_t [ngpu];

    //init data for gpu d   ///   round robine 

   	int num_threads = 128;
    	int num_blocks = ceil ((double)nnzTR / (double)num_threads) ; 

	int  * start_col  = new int[ngpu];
	int  * end_col    = new int[ngpu];
	int  * start_nnz  = new int[ngpu];	
  int  * start_x    = new int[ngpu];
	int * dev_m            = new int      [ngpu];
	int * dev_n            = new int      [ngpu];
	int * dev_nnz          = new int      [ngpu];
	int ** host_cscColPtr  = new int    * [ngpu];
	int ** dev_cscColPtrTR   = new int    * [ngpu];
	int * dev_cscRowIdxTR;
	VALUE_TYPE ** dev_cscValTR   = new double * [ngpu];
	VALUE_TYPE ** dev_x = new double * [ngpu];
	VALUE_TYPE ** dev_b = new double * [ngpu];

	
   int balance_nnz;
  balance_nnz = floor( nnzTR / ngpu);
        
	for (int d = 0; d < ngpu; d++){

		cudaSetDevice(d);

    //dynamic col
     
    if (d == 0){
    start_col[d] = 0;}
    else {
    start_col[d] = end_col[d-1]+1;
    }
    
     // init end_col
    
    end_col[d] = start_col[d];
    
    do{
    if (end_col[d] < m){
    end_col[d]++;
    }else{
    break;
    }
    }
    while((cscColPtrTR[end_col[d]] - cscColPtrTR[start_col[d]])< balance_nnz);
    
    end_col[d]--;
    

		dev_n[d]   = end_col[d] - start_col[d] + 1   ;                         // n col in device d
		dev_m[d]   = m;                                                     // m row 

		long long nnzTR_ll = cscColPtrTR[end_col[d] + 1] - cscColPtrTR[start_col[d]];     // nonzero 
		//long long matrix_data_space = nnzTR_ll * sizeof(double) + 
										//nnzTR_ll * sizeof(int) + 
										//(long long)(dev_n[d]+1) * sizeof(int) + 
										//(long long)dev_m[d] * sizeof(double) +
										//(long long)dev_n[d] * sizeof(double);
		//double matrix_size_in_gb = (double)matrix_data_space / 1e9;
		

		dev_nnz[d] = (int)(cscColPtrTR[end_col[d] + 1] - cscColPtrTR[start_col[d]]);           // nonzero 
   
    if (d == 0){
      start_nnz[d] = 0;
      start_x[d] = 0;
      }
    else{
      start_nnz[d] = start_nnz[d-1] + dev_nnz[d-1];
      start_x[d] = start_x[d-1]+dev_n[d-1];
    }
    
		host_cscColPtr[d] = new int[dev_n[d] + 1];                                             // host colPTR
		for (int i = 0; i < dev_n[d] + 1; i++) {
			host_cscColPtr[d][i] = (int)(cscColPtrTR[start_col[d] + i] - cscColPtrTR[start_col[d]]);
		//	printf("device %d, col %d has %d nnz\n",d,i,host_cscColPtr[d][i]);
		}
                printf("start from %d value, has %d nnz for device %d\n", start_nnz[d], dev_nnz[d], d);

	}


    // transfer host mem to device mem
   // int *d_cscColPtrTR;
   // int *d_cscRowIdxTR;
    //VALUE_TYPE *d_cscValTR;
   // VALUE_TYPE *d_b;
   // VALUE_TYPE *d_x;
    
    // transfer host mem to device mem


   cudaMallocManaged((void **)&dev_cscRowIdxTR, nnzTR * sizeof(int));
   //cudaMemcpy(dev_cscRowIdxTR, cscRowIdxTR, nnzTR * sizeof(int),  cudaMemcpyHostToDevice);
   for (int i = 0; i < nnzTR; i++){
	dev_cscRowIdxTR[i] = cscRowIdxTR[i];
	}

    for (int d = 0; d < ngpu; d++){
		cudaSetDevice(d);
		//cudaStreamCreate(&(stream[d]));

                cudaDeviceProp deviceProp;
   		cudaGetDeviceProperties(&deviceProp, d);
   		printf("---------------------------------------------------------------------------------------------\n");
   		printf("Device [ %i ] %s @ %4.2f MHz\n", d, deviceProp.name, deviceProp.clockRate * 1e-3f);
                //cudaStreamCreate(&(stream[d]));
    		// Matrix L
    		cudaMalloc((void **)&dev_cscColPtrTR[d], (size_t)((dev_n[d]+1) * sizeof(int)));
    		
    		cudaMalloc((void **)&dev_cscValTR[d],    dev_nnz[d]  * sizeof(VALUE_TYPE));

    		err[d] = cudaMemcpy(dev_cscColPtrTR[d], host_cscColPtr[d], (size_t)((dev_n[d]+1) * sizeof(int)),   cudaMemcpyHostToDevice);
    		
		if (err[d] != cudaSuccess){
			printf("dev_cscColPtrTr faild at device %d\n",d);
                        return 1;

   		}

    		err[d] = cudaMemcpy(dev_cscValTR[d],    &cscValTR[cscColPtrTR[start_col[d]]],    (size_t)(dev_nnz[d] * sizeof(VALUE_TYPE)),   cudaMemcpyHostToDevice);
                

                if (err[d] != cudaSuccess){
                        printf("dev_cscValTr faild at device %d\n",d);
                        return 1;

                }

                printf("mem allocate Matrix L\n");

    		// Vector b

   		 cudaMalloc((void **)&dev_b[d], dev_n[d] * rhs * sizeof(VALUE_TYPE));
   		err[d] = cudaMemcpy(dev_b[d], &b[start_col[d]], (size_t)(dev_n[d] * rhs * sizeof(VALUE_TYPE)), cudaMemcpyHostToDevice);
		printf("the start x: %d in device %d has value in b is %f \n",start_col[d], d, b[start_col[d]]);

		if (err[d] != cudaSuccess){
                        printf("dev_b faild at device %d\n",d);
                        return 1;

                }
		printf("mem allocate B\n");

    		// Vector x
    		cudaMalloc((void **)&dev_x[d], (size_t)(dev_n[d] * rhs * sizeof(VALUE_TYPE)));
    		err[d] = cudaMemset(dev_x[d], 0, (size_t)(dev_n[d] * rhs * sizeof(VALUE_TYPE)));
                if (err[d] != cudaSuccess){
                        printf("dev_x faild at device %d\n",d);
                        return 1;

                }
                printf("mem set x\n");

		}	

     printf("Memcpy from Host to Device seccuss"); 

    //  - cuda syncfree SpTRSV analysis start!
    printf(" - cuda syncfree SpTRSV analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);




    // malloc tmp memory to generate in-degree
    //int **d_graphInDegree = new int * [ngpu];
    int *global_graphInDegree_backup = (int *)malloc(sizeof(int) * m);
    int *global_graphInDegree;
     
  //  cudaMallocManaged((void **)&global_graphInDegree_backup, (size_t)(m * sizeof(int)));
    cudaMallocManaged((void **)&global_graphInDegree, (size_t)(m * sizeof(int)));
    //for (int d = 0; d < ngpu; d++){
//	cudaSetDevice(d);
//	cudaMalloc((void **)&d_graphInDegree[d], (size_t)(dev_n[d] * sizeof(int)));
//        cudaMalloc((void **)&d_graphInDegree_backup[d], (size_t)(dev_n[d] * sizeof(int)));
//	} 


    for (int i=0; i < BENCH_REPEAT; i++){
    

	cudaMemset(global_graphInDegree, 0, (size_t)(m * sizeof(int)));

    	for (int d = 0; d < ngpu ; d++){
	      	cudaSetDevice(d);
          num_blocks = ceil ((double)dev_nnz[d] / (double)num_threads);
	      	sptrsv_syncfree_cuda_analyser<<< num_blocks, num_threads >>>
                                      (dev_cscRowIdxTR, m,  dev_nnz[d], global_graphInDegree, start_nnz[d],d);
	       }

    	for (int d = 0; d < ngpu; d++){
	      	cudaSetDevice(d);
	      	cudaDeviceSynchronize();
          //cudaDeviceSynchronize();
        	}
    }
    
    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("cuda syncfree SpTRSV analysis on L used %4.2f ms\n", time_cuda_analysis);

    //  - cuda syncfree SpTRSV solve start!
    printf(" - cuda syncfree SpTRSV solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *left_sum;

    cudaMallocManaged((void **)&left_sum, (size_t)(sizeof(VALUE_TYPE) * m * rhs));

	// backup in-degree array, only used for benchmarking multiple runs
    cudaError_t err2;
    err2 =  cudaMemcpy(global_graphInDegree_backup, global_graphInDegree, (size_t)(m * sizeof(int)), cudaMemcpyDeviceToHost);
    if (err2 != cudaSuccess){
	printf("copy to backup failed\n");
    }
     
    //  for (int i = 0; i< m; i++){
    //    printf(" x %d has degree %d\n", i , global_graphInDegree_backup[i]);
     
    // }
      VALUE_TYPE **dev_left_sum = new VALUE_TYPE * [ngpu];
    int **dev_in_degree = new int * [ngpu];

    for (int d = 0; d < ngpu ; d++){
      cudaSetDevice(d);
  	 cudaMalloc((void **)&dev_left_sum[d], (size_t)(dev_n[d] * sizeof(VALUE_TYPE)));
   	 cudaMemset(dev_left_sum[d], 0, (size_t)(dev_n[d] * sizeof(VALUE_TYPE)));
     cudaMalloc((void **)&dev_in_degree[d], (size_t)(dev_n[d] * sizeof(int)));
   	 cudaMemset(dev_in_degree[d], 0, (size_t)(dev_n[d] * sizeof(int))); 
   	 
    }
    
    // this is for profiling while loop only
    int **dev_while_profiler = new int * [ngpu];
    int **while_profiler = new int * [ngpu];

    for (int d = 0; d < ngpu ; d++){
         cudaSetDevice(d);
    	 cudaMalloc((void **)&dev_while_profiler[d], (size_t)(dev_n[d] * sizeof(int)));
   	 cudaMemset(dev_while_profiler[d], 0, (size_t)(dev_n[d] * sizeof(int)));
   	 while_profiler[d] = (int *)malloc(sizeof(int) * dev_n[d]); 
    }

    // step 5: solve L*y = x
    double time_cuda_solve = 0;
    //gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        int *global_graphInDegree_r;		
	cudaMemset(dev_left_sum, 0, (size_t)(sizeof(VALUE_TYPE) * m * rhs));
        cudaMallocManaged((void **)&global_graphInDegree_r, (size_t)(m * sizeof(int)));

	// get a unmodified in-degree array, only for benchmarking use
       	err2 =  cudaMemcpy(global_graphInDegree_r, global_graphInDegree_backup, (size_t)(m * sizeof(int)), cudaMemcpyHostToDevice);
       	if (err2 != cudaSuccess){
       //		 printf("backup failed\n");
	   	 }


	 // clear left_sum array, only for benchmarking use
        for (int d =0; d < ngpu; d++ ){	
		cudaSetDevice(d); 
       		 cudaMemset(dev_x[d], 0, (size_t)(dev_n[d] * rhs * sizeof(VALUE_TYPE)));
	}

        gettimeofday(&t1, NULL);

        //solve
        for (int d = 0 ; d < ngpu; d++){
		cudaSetDevice(d);
      		  if (rhs == 1)
       		 {
        	    num_threads = WARP_PER_BLOCK * WARP_SIZE;
        	    num_blocks = ceil (((double)dev_n[d] / (double)(num_threads/WARP_SIZE)));
              //printf("%d\n" ,d); 
              int device_id = d;  
        	    sptrsv_syncfree_cuda_executor<<< num_blocks, num_threads >>>
                                         (dev_cscColPtrTR[d], dev_cscRowIdxTR, dev_cscValTR[d],
                                          global_graphInDegree_r, left_sum, dev_in_degree[d], dev_left_sum[d],
                                          dev_n[d], substitution, dev_b[d], dev_x[d], dev_while_profiler[d], start_nnz[d], start_x[d], device_id);
       		 }
       		 else
       		 {
        	    num_threads = 4 * WARP_SIZE;
        	    num_blocks = ceil ((double) dev_n[d] / (double)(num_threads/WARP_SIZE));
        	    sptrsm_syncfree_cuda_executor<<< num_blocks, num_threads >>>
					(dev_cscColPtrTR[d], dev_cscRowIdxTR, dev_cscValTR[d],
                                          global_graphInDegree, left_sum,
                                          dev_n[d], substitution, rhs, opt, dev_b[d], dev_x[d], dev_while_profiler[d]);
        	}

	   }
       

         for (int d =0; d < ngpu; d++){

             cudaSetDevice(d);
	     cudaDeviceSynchronize();
	 }

 	   gettimeofday(&t2, NULL);

	    time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	}
    time_cuda_solve /= BENCH_REPEAT;
    double flop = 2*(double)rhs*(double)nnzTR;

    printf("cuda syncfree SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, flop/(1e6*time_cuda_solve));
    *gflops = flop/(1e6*time_cuda_solve);

    //copy x
    for (int d =0 ; d < ngpu; d++){
         cudaSetDevice(d);
   	 cudaMemcpy(&x[start_col[d]], dev_x[d], (size_t)(dev_n[d] * rhs * sizeof(VALUE_TYPE)), cudaMemcpyDeviceToHost);
 	
	 printf("the start x: %d in device %d has value %f, real value should be %f \n",start_col[d], d, x[start_col[d]],x_ref[start_col[d]]);
   
    }


    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
    //    if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("cuda syncfree SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("cuda syncfree SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

	long long unsigned int while_count = 0;
    // profile while loop
    for (int d = 0 ; d < ngpu; d++){
    	cudaMemcpy(while_profiler[d], dev_while_profiler[d], (size_t)(dev_n[d] * sizeof(int)), cudaMemcpyDeviceToHost);

   	 for (int i = 0; i < dev_n[d]; i++)
    		{
        	while_count += while_profiler[d][i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    		}
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);
     }
    
// step 6: free resources
	cudaFree(left_sum);
	cudaFree(dev_cscRowIdxTR);

	cudaFree(global_graphInDegree);
	cudaFree(global_graphInDegree_backup);

    for (int d = 0 ; d < ngpu; d++){
    	free(while_profiler[d]);
      cudaFree(dev_left_sum[d]);
      cudaFree(dev_in_degree[d]);
    	cudaFree(dev_while_profiler[d]);
    	cudaFree(dev_cscColPtrTR[d]);
    	cudaFree(dev_cscValTR[d]);
    	cudaFree(dev_b[d]);
    	cudaFree(dev_x[d]);
    }

    return 0;
}

#endif



