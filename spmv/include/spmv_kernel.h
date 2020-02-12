#ifndef SPMV_KERNEL
#define SPMV_KERNEL

/*
int csr5_kernel(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y);
*/

int spMV_mgpu_baseline(int m, int n, long long nnz, double * alpha,
				 double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				 double * x, double * beta,
				 double * y,
				 int ngpu);
int spMV_mgpu_v1(int m, int n, long long nnz, double * alpha,
				  double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu,
				  int kernel);

int spMV_mgpu_v2(int m, int n, long long nnz, double * alpha,
				  double * csrVal, long long * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y,
				  int ngpu, 
				  int kernel,
				  long long nb,
				  int copy_of_workspace);

int get_row_from_index(int n, long long * a, long long idx);

double get_time();

double get_gpu_availble_mem(int ngpu);


#endif /* SPMV_KERNEL */
