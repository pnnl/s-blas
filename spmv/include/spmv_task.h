#ifndef SPMV_TASK
#define SPMV_TASK

struct spmv_task
{
	int dev_id;

	long long start_idx;
	long long end_idx;
	int start_row;
	int end_row;
	bool start_flag;
	bool end_flag;

	int * host_csrRowPtr;
	int * host_csrColIndex;
	double * host_csrVal;
	double * host_x;
	double * host_y;
	double y2;

	double * local_result_y;

	cusparseOperation_t trans;
	int 				dev_m;
	int 				dev_n;
	int 				dev_nnz;
	double * 			alpha;
	cusparseMatDescr_t 	descr;
	double *			dev_csrVal;
	int *				dev_csrRowPtr;
	int *				dev_csrColIndex;
	double *		 	dev_x;
	double *			beta;
	double *			dev_y;
};

#endif /* SPMV_TASK */