#ifndef SPMV_TASK_GPU
#define SPMV_TASK_GPU

struct spmv_task_GPU
{
	int dev_id;

	int start_idx;
	int end_idx;
	int start_row;
	int end_row;
	bool start_flag;
	bool end_flag;

	int * master_csrRowPtr;
	int * master_csrColIndex;
	double * master_csrVal;
	double * master_x;
	double * mster_y;
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

#endif /* SPMV_TASK_GPU */