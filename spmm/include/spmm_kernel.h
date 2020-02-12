#ifndef SPMM_KERNEL
#define SPMM_KERNEL

//int SpMM_cpu_openblas_baseline(int M, int N, int K, double * alpha, double * A, double * B, double * beta, double * C);

int cusparse_mgpu_csrmm(const int m,
			const int n,
			const int k,
                        const double * alpha,
			const int nnz_A,
			int * csrRowPtr_A,
			int * csrColIndex_A,
			double * csrVal_A,
			const double * beta,
			double * B_dense,
			double * C_dense,
			const int ngpu);


int cusparse_mgpu_csrmm_omp(const int m,
			const int n,
			const int k,
                        const double * alpha,
			const int nnz_A,
			int * csrRowPtr_A,
			int * csrColIndex_A,
			double * csrVal_A,
			const double * beta,
			double * B_dense,
			double * C_dense,
			const int ngpu);




#endif /* SPMM_KERNEL */
