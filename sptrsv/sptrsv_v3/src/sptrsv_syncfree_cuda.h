#ifndef _SPTRSV_SYNCFREE_CUDA_
#define _SPTRSV_SYNCFREE_CUDA_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>



//calculated the local indegree in the shmem
__global__
void sptrsv_syncfree_cuda_analyser(const int   *dev_cscRowIdx,
                                   int   *shmem_graphInDegree,
                                   const int    dev_nnz         
                                    )
{
    const int device_id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (device_id < dev_nnz)
    {
        atomicAdd(&shmem_graphInDegree[dev_cscRowIdx[device_id]], 1);
        
    }
}


//calculate the overall degree for each component. implemented as a reducion function across all GPUs  
//not be used in the final implementation
/*
__global__
void sptrsv_syncfree_cuda_degree(  int   *shmem_graphInDegree,
                                   int   *dev_in_degree,
                                   const int  dev_n,
                                   const int  start_x,
                                   const int   np            
                                    )
{
    const int device_id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (device_id < dev_n)
    {    
       int tempdata = 0;
       for (int n=0; n<np; n++){
        nvshmem_int_get(&tempdata, shmem_graphInDegree+start_x+device_id, sizeof(int), n );    
        dev_in_degree[device_id] = tempdata + dev_in_degree[device_id];
       }     
    }
}

*/                                        

//the SpTRSV solver. it is splited into two phases:
//the lock waiting phase for degree check
//the solver and update phase for x and indegree update
//we used the nvshmem_get API collect data   
__global__
void sptrsv_syncfree_cuda_executor(const int* __restrict__        d_cscColPtr,
                                   const int* __restrict__        d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int*                     shmem_graphInDegree,
                                         int*                     dev_in_degree,
                                         VALUE_TYPE*              shmem_left_sum,
                                         VALUE_TYPE*              dev_left_sum,
                                   const int                      d_n,
                                   const int                      substitution,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                   const int                      start_nnz,
                                   const int                      start_x,
				                           const int			                d_id,
                                   const int                      np)
					
{
 
const int device_id = blockIdx.x * blockDim.x + threadIdx.x;  
int device_x_id =  device_id / WARP_SIZE;
if (device_x_id < d_n) {      
    device_x_id = substitution == SUBSTITUTION_FORWARD ? 
                  device_x_id : d_n - 1 - device_x_id;   //get x id
                  
    

    int starting_x = start_x;
    
    starting_x = substitution == SUBSTITUTION_FORWARD ? 
                  starting_x : d_n - 1 - starting_x;
                  
    int global_x_id = device_x_id + starting_x;
   
    
    volatile __shared__ int s_graphInDegree[WARP_PER_BLOCK];    //using sharememory for block
    volatile __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK];

    // Initialize

    const int local_warp_id = threadIdx.x / WARP_SIZE;  // block x id
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
   
  
    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                    d_cscColPtr[device_x_id] : d_cscColPtr[device_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos]; // get x id in device to get d_cscval
    
    
    if (threadIdx.x < WARP_PER_BLOCK) { s_graphInDegree[threadIdx.x] = 0; s_left_sum[threadIdx.x] = 0; }
    __syncthreads;
  
  
    int x_in_degree =0;
    int tempdata = 0;   
    
    //lock-waiting
            
    do {    
        x_in_degree =0;
        
        //using a loop
        
        for (int n=0; n<np; n++){
            nvshmem_int_get_nbi(&tempdata, shmem_graphInDegree+global_x_id, sizeof(int), n );
            x_in_degree = tempdata + x_in_degree;
        }
       
      //warp level parallism
      // after comparing performance between loop and parallel access, loop outperformance than parallsim inside while due to wrap scheduler       
    //  if (lane_id < np)
    //     nvshmem_int_get_nbi(&x_in_degree, shmem_graphInDegree+global_x_id, sizeof(int), lane_id );  // for each lane, get one value
   
    //  for (int offset = np; offset > 0; offset /= 2)
    //    x_in_degree += __shfl_down_sync(0xffffffff, x_in_degree, offset);
      
     
    //   x_in_degree = __shfl_sync(0xffffffff, x_in_degree, 0);
                                 
    }
     while ( x_in_degree!= (s_graphInDegree[local_warp_id]+ dev_in_degree[device_x_id] +1));

    
    // solve x
    VALUE_TYPE templeft = 0;
    
    //warp level parallelism 
    
    if (lane_id < np)
     nvshmem_double_get(&templeft, shmem_left_sum+global_x_id, sizeof(VALUE_TYPE), lane_id );   // for each lane, get one value
   
    for (int offset = np+1; offset > 0; offset /= 2)
     templeft += __shfl_down_sync(0xffffffff, templeft, offset);
     
     
    templeft = __shfl_sync(0xffffffff, templeft, 0);
     
    VALUE_TYPE xi = s_left_sum[local_warp_id] + dev_left_sum[device_x_id]+templeft;// + shmem_left_sum[global_x_id];
   
   
   //using a loop
   //warp level has small number of loop by using __shlf_down_sync
  // VALUE_TYPE xi = s_left_sum[local_warp_id] + dev_left_sum[device_x_id];    
  // for (int n=0; n<np; n++){
  //     nvshmem_double_get(&templeft, shmem_left_sum+global_x_id, sizeof(VALUE_TYPE), n );      
   //  xi = templeft + xi;
   //  }
       
    xi = (d_b[device_x_id] - xi) * coef;
    



    // update intermediate value
    
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[device_x_id] : d_cscColPtr[device_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[device_x_id+1] : d_cscColPtr[device_x_id+1]-1;

    
    for (int jj = start_ptr+lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);

        const int nnzID = j;
        const int rowIdx = d_cscRowIdx[nnzID];
        
        //  printf("processing %d nnz %d lane for x %d\n", nnzID, rowIdx, global_x_id);
         //  printf("processing %d nnz %d lane for x %d\n", nnzID, rowIdx, global_x_id);
        const bool cond_inBlock = substitution == SUBSTITUTION_FORWARD ? 
                    (rowIdx < starting_x + WARP_PER_BLOCK) : (rowIdx > starting_x - WARP_PER_BLOCK);   
        const bool cond_inDevice = substitution == SUBSTITUTION_FORWARD?
                    (rowIdx < starting_x + d_n) : (rowIdx > starting_x - d_n); 
                       
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
            
             atomicAdd((VALUE_TYPE *)&shmem_left_sum[rowIdx], xi * d_cscVal[j]);
             __threadfence();                     
                        
            atomicSub((int *)&shmem_graphInDegree[rowIdx], 1);
                
        }
    }

    //finish
    if (!lane_id) d_x[device_x_id] = xi;
   }
}




// worklaod distribution function
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
			                         int     	     ngpu,
                               int           task)
{


     nvshmem_init();
     int mype;
     int np;
     mype = nvshmem_my_pe();
     np = nvshmem_n_pes();  // np total number of pe
     
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    cudaError_t err;

    //init local data for gpu np   ///   round robine 

   	int num_threads = 512;
   	int num_blocks;

//host matrix
	int  * start_col  = new int[np*task];
	int  * end_col    = new int[np*task];
	int  * start_nnz  = new int[np*task];	
  int  * start_x    = new int[np*task];
 // int  * dev_start_x     = new int[np*task];
	int * dev_m            = new int      [np*task];
	int * dev_n            = new int      [np*task];
	int * dev_nnz          = new int      [np*task];
	int ** host_cscColPtr  = new int    * [np*task];
 
 //initial device matrix
	int ** dev_cscColPtrTR = new int    * [task];
  int ** dev_cscRowIdxTR = new int    * [task];  
	VALUE_TYPE ** dev_cscValTR = new VALUE_TYPE    * [task];
	VALUE_TYPE ** dev_x = new VALUE_TYPE    * [task];
	VALUE_TYPE ** dev_b = new VALUE_TYPE    * [task];
 
  //init local degree for each x;
  int **dev_in_degree = new int * [task];
	
   int balance_nnz;
  balance_nnz = floor( nnzTR / (np*task));
   
   
   // data distribution 
   
   // balance for n
  for (int d = 0; d < (np*task); d++){
 
		start_col[d] = floor((d)     * m / (np*task));                           //first x in task d
		end_col[d]   = floor((d + 1) * m / (np*task)) - 1;                       //last x in task d

		dev_n[d]   = end_col[d] - start_col[d] + 1;                         // n col in task d
		dev_m[d]   = m;                                                     // m row 

		//long long nnzTR_ll = cscColPtrTR[end_col[d] + 1] - cscColPtrTR[start_col[d]];     // nonzero 
		//long long matrix_data_space = nnzTR_ll * sizeof(double) +                         //calculate matrix size in task d
		//								nnzTR_ll * sizeof(int) +                                          //using for check if the data could be fit into GPU memory
		//								(long long)(dev_n[d]+1) * sizeof(int) + 
		//								(long long)dev_m[d] * sizeof(double) +
		//								(long long)dev_n[d] * sizeof(double);
		//double matrix_size_in_gb = (double)matrix_data_space / 1e9;
		

		dev_nnz[d] = (int)(cscColPtrTR[end_col[d] + 1] - cscColPtrTR[start_col[d]]);           // nonzero 
    if (d == 0){
      start_nnz[d] = 0;
      start_x[d] = 0;
      }
    else{
      start_nnz[d] = start_nnz[d-1] + dev_nnz[d-1];      //calculate the start point in sparse matrix
      start_x[d] = start_x[d-1]+dev_n[d-1];
    }
    
		host_cscColPtr[d] = new int[dev_n[d] + 1];                                             // host colPTR
		for (int i = 0; i < dev_n[d] + 1; i++) {
			host_cscColPtr[d][i] = (int)(cscColPtrTR[start_col[d] + i] - cscColPtrTR[start_col[d]]);
		}
      if (mype == 0)   printf("start from %d value, has %d nnz for device %d\n", start_nnz[d], dev_nnz[d], d);

}
 
      // balance for nnz //not used for performance
   /*  
	for (int d = 0; d < (np*task); d++){

	    //int dev = d%ngpu;
		  //cudaSetDevice(dev);    // distribute task d to device dev in RR

    //dynamic col
     
    if (d == 0){
    start_col[d] = 0;}
    else {
    start_col[d] = end_col[d-1]+1;
    }
    
     // init end_col
    // balance nnz for each tasks (didn't break the col)
    
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
    
		dev_n[d]   = end_col[d] - start_col[d]+1   ;                         // n col in device d
		dev_m[d]   = m;                                                     // m row 
		
		dev_nnz[d] = (int)(cscColPtrTR[end_col[d] + 1] - cscColPtrTR[start_col[d]]);           // nonzero 
   
     //record nnz pointer for reconstruction
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
     
    if (mype == 0)    
                printf("start from %d value, has %d nnz for task %d\n", start_nnz[d], dev_nnz[d], d);

	}

*/

   // copy loctal deta to gpu np
  
		  cudaSetDevice(mype);    // distribute task d to device dev in RR
      
      
      cudaDeviceProp deviceProp;
   		cudaGetDeviceProperties(&deviceProp, mype);
   
   		printf("---------------------------------------------------------------------------------------------\nDevice [ %i ] %s @ %4.2f MHz\n", mype, deviceProp.name, deviceProp.clockRate * 1e-3f);
                   
    for (int t =0 ; t<task; t++)
    {
           
    		// Matrix L
   		cudaMalloc((void **)&dev_cscColPtrTR[t], (size_t)((dev_n[mype+t*np]+1) * sizeof(int)));
    		
   		cudaMalloc((void **)&dev_cscValTR[t],    dev_nnz[mype+t*np]  * sizeof(VALUE_TYPE));
      
      cudaMalloc((void **)&dev_cscRowIdxTR[t],    dev_nnz[mype+t*np]  * sizeof(int));

      err = cudaMemcpyAsync(dev_cscColPtrTR[t], host_cscColPtr[mype+t*np], (size_t)((dev_n[mype+t*np]+1) * sizeof(int)),   cudaMemcpyHostToDevice);
    		
		  if (err != cudaSuccess){
		      	printf("dev_cscColPtrTr faild at device %d\n",mype);
            return 1;
   		}

      err = cudaMemcpyAsync(dev_cscValTR[t],    &cscValTR[cscColPtrTR[start_col[mype+t*np]]],    (size_t)(dev_nnz[mype+t*np] * sizeof(VALUE_TYPE)),   cudaMemcpyHostToDevice);
      
      if (err != cudaSuccess){
            printf("dev_cscValTr faild at device %d\n",mype);
            return 1;
      }
      
      
      err=cudaMemcpyAsync(dev_cscRowIdxTR[t], &cscRowIdxTR[cscColPtrTR[start_col[mype+t*np]]], dev_nnz[mype+t*np] * sizeof(int),  cudaMemcpyHostToDevice);
       if (err != cudaSuccess){
		      	printf("dev_cscRowIdxTR faild at device %d\n",mype);
            return 1;
   		}
      
      //printf("mem allocate Matrix L\n");

       // Vector b

      cudaMalloc((void **)&dev_b[t], dev_n[mype+t*np] * rhs * sizeof(VALUE_TYPE));
   		err = cudaMemcpyAsync(dev_b[t], &b[start_col[mype+t*np]], (size_t)(dev_n[mype+t*np] * sizeof(VALUE_TYPE)), cudaMemcpyHostToDevice);
		 // printf("the start x: %d in device %d has value in b is %f \n",start_col[mype+t*np], mype, b[start_col[mype+t*np]]);

      if (err != cudaSuccess){
            printf("dev_b faild at device %d\n",mype);
            return 1;
      }
		 // printf("mem allocate B\n");

      // Vector x
    	cudaMalloc((void **)&dev_x[t], (size_t)(dev_n[mype+t*np] * sizeof(VALUE_TYPE)));
    	err = cudaMemsetAsync(dev_x[t], 0, (size_t)(dev_n[mype+t*np] * sizeof(VALUE_TYPE)));
     
      if (err != cudaSuccess){
           printf("dev_x faild at device %d\n", mype);
           return 1;
      }
     // printf("mem set x\n");
     
       //loacl dev_in_deree
      cudaMalloc((void **)&dev_in_degree[t], (size_t)(dev_n[mype+t*np] * sizeof(int)));
      err=cudaMemsetAsync(dev_in_degree[t], 0, (size_t)(dev_n[mype+t*np] * sizeof(int)));
      if (err != cudaSuccess){
		      	printf("dev_in_Degree faild at device %d\n",mype);
            return 1;
   		} 
     
   }	

     printf("Memcpy from Host to Device %d seccuss\n", mype); 
     
    
// inital shemem data


    
    int *  shmem_graphInDegree;  // update InDegree inside
         
      shmem_graphInDegree = (int *)nvshmem_malloc(n * sizeof(int));
      
      err=cudaMemsetAsync(shmem_graphInDegree, 0, n * sizeof(int));
       if (err != cudaSuccess){
		      	printf("shmem_graphInDegree faild at device %d\n",mype);
            return 1;
   		}
     



    //  - cuda syncfree SpTRSV analysis start!
     if (mype == 0)  printf(" - cuda syncfree SpTRSV analysis start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
   

   for (int t = 0; t < task ; t++){
	  
     num_blocks = ceil ((double)dev_nnz[mype+t*np] / (double)num_threads);
   	 sptrsv_syncfree_cuda_analyser<<< num_blocks, num_threads >>>
             (dev_cscRowIdxTR[t], shmem_graphInDegree, dev_nnz[mype+t*np]);  
             
             
     }  
  
	    cudaDeviceSynchronize();
     
    
    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("device:%d cuda syncfree SpTRSV analysis on L used %4.2f ms\n", mype, time_cuda_analysis);


   //  - cuda syncfree SpTRSV solve start!
    if (mype == 0)  printf(" - cuda syncfree SpTRSV solve start!\n");
    
    // Malloc local left_sum for x
   VALUE_TYPE **dev_left_sum = new VALUE_TYPE * [task];

   for (int t = 0; t < task ; t++){

     cudaMalloc((void **)&dev_left_sum[t], (size_t)(dev_n[mype+t*np] * sizeof(VALUE_TYPE)));
     err=cudaMemsetAsync(dev_left_sum[t], 0, (size_t)(dev_n[mype+t*np] * sizeof(VALUE_TYPE)));
       
     if (err != cudaSuccess){
		      	printf("dev_left_sum faild at device %d\n",mype);
            return 1;
   		}
   }
   
   //Malloc shemem left_sum for x;     
   VALUE_TYPE *  shmem_left_sum;  // update left_sum inside
    
   shmem_left_sum = (VALUE_TYPE *)nvshmem_malloc(n * sizeof(VALUE_TYPE));    
   err=cudaMemsetAsync(shmem_left_sum, 0, n * sizeof(VALUE_TYPE));
   if (err != cudaSuccess){
  	   printf("shmem_left_sum faild at device %d\n",mype);
       return 1;
 		}
    
  
    double time_cuda_solve = 0;
    //solve 
    gettimeofday(&t1, NULL);
    
    for (int t = 0; t < task ; t++){
    
      num_threads = WARP_PER_BLOCK * WARP_SIZE;
      num_blocks = ceil (((double)dev_n[mype+t*np] / (double)(num_threads/WARP_SIZE)));
              
      sptrsv_syncfree_cuda_executor<<< num_blocks, num_threads >>>
                                         (dev_cscColPtrTR[t], dev_cscRowIdxTR[t], dev_cscValTR[t],
                                          shmem_graphInDegree, dev_in_degree[t], shmem_left_sum, dev_left_sum[t],
                                          dev_n[mype+t*np], substitution, dev_b[t], dev_x[t], start_nnz[mype+t*np], start_x[mype+t*np], mype, np);
                                          
   
    }
                                          
    cudaDeviceSynchronize();
    
     gettimeofday(&t2, NULL);

     time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
     double flop = 2*(double)rhs*(double)nnzTR;

    printf("device:%d --\n", mype);

    printf("cuda syncfree SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n", time_cuda_solve, flop/(1e6*time_cuda_solve));
    *gflops = flop/(1e6*time_cuda_solve);
    
     
     
      //copy x
  
    for (int t = 0; t < task ; t++){   
      cudaMemcpyAsync(&x[start_col[mype+t*np]], dev_x[t], (size_t)(dev_n[mype+t*np] * sizeof(VALUE_TYPE)), cudaMemcpyDeviceToHost);
    }


    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int t = 0; t < task ; t++){
      for (int i = 0; i < dev_n[mype]; i++)
      {
        ref += abs(x_ref[(i+start_col[mype+t*np])]);
        res += abs(x[(i+start_col[mype+t*np])] - x_ref[(i+start_col[mype+t*np])]);
      }
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("device:%d cuda syncfree SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", mype,res);
    else
        printf("device:%d cuda syncfree SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", mype, res);
        
       
    
//  free resources

     	nvshmem_free(shmem_graphInDegree);
      nvshmem_free(shmem_left_sum);
      
      for (int t = 0; t < task ; t++){ 
        cudaFree(dev_left_sum[t]);
        cudaFree(dev_in_degree[t]);
      	cudaFree(dev_cscColPtrTR[t]);
    	  cudaFree(dev_cscValTR[t]);
        cudaFree(dev_cscRowIdxTR[t]);
    	  cudaFree(dev_b[t]);
    	  cudaFree(dev_x[t]);
       }
   
    
    nvshmem_finalize();
   
    return 0;
}

#endif



