#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "mmio.h"
#include <float.h>
#include <omp.h>
//#include "anonymouslib_cuda.h"
#include <cuda_profiler_api.h>
#include "spmv_kernel.h"
#include <limits>
using namespace std;



void print_error(cusparseStatus_t status) {
	if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
		cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << endl;
	else if (status == CUSPARSE_STATUS_ALLOC_FAILED)
		cout << "CUSPARSE_STATUS_ALLOC_FAILED" << endl;
	else if (status == CUSPARSE_STATUS_INVALID_VALUE)
		cout << "CUSPARSE_STATUS_INVALID_VALUE" << endl;
	else if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
		cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << endl;
	else if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
		cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << endl;
	else if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
		cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << endl;
}



int main(int argc, char *argv[]) {


	if (argc < 6) {
		cout << "Incorrect number of arguments!" << endl;
		cout << "Usage ./spmv [input matrix file] [number of GPU(s)] [number of test(s)] [kernel version (1-3)] [data type ('f' or 'b')]"  << endl;
		return -1;
	}

	char input_type = argv[1][0];

	char * filename = argv[2];

	int ngpu = atoi(argv[3]);
	int repeat_test = atoi(argv[4]);
	int kernel_version = atoi(argv[5]);
	
	//int divide = atoi(argv[7]);
	//int copy_of_workspace = atoi(argv[8]);

	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int m, n;
    long long nnz;   
    int * cooRowIndex;
    int * cooColIndex;
    double * cooVal;
    long long * csrRowPtr;


    int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount < ngpu) {
		cout << "Error: Not enough number of GPUs. Only " << deviceCount << "available." << endl;
		return -1;
	}
	if (ngpu <= 0) {
		cout << "Error: Number of GPU(s) needs to be greater than 0." << endl;
		return -1;
	}

	if (kernel_version != 1 && kernel_version != 2 && kernel_version != 3) {
		cout << "Error: The kernel version can only be: 1, 2, or 3." << endl;
		return -1;
	}

	// if (divide <= 0) {
	// 	cout << "Error: Number of tasks needs to be greater than 0." << endl;
	// 	return -1;
	// }

	// if (copy_of_workspace <= 0) {
	// 	cout << "Error: Number of Hyper-Q needs to be greater than 0." << endl;
	// 	return -1;
	// }



	cout << "Using " << ngpu << " GPU(s)." << endl; 
	cout << "Kernel #" << kernel_version << " is selected." << endl;
	//cout << divide <<  "total task(s) will be generated for version 2 with "<< copy_of_workspace << " Hyper-Q(s) on each GPU." << endl;

    if (input_type == 'f') {

	    cout << "Loading input matrix from " << filename << endl;
	    if ((f = fopen(filename, "r")) == NULL) {
	        exit(1);
	    }
	    if (mm_read_banner(f, &matcode) != 0) {
	        printf("Could not process Matrix Market banner.\n");
	        exit(1);
	    }
	    int nnz_int;
	    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz_int)) !=0) {
	        exit(1);
	    }
	    nnz = nnz_int;
	    cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
	    cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));
	   char data_type = argv[6][0];
	    // Read matrix from file into COO format
	    for (int i = 0; i < nnz; i++) {
	    	if (data_type == 'b') { // binary input
	    		fscanf(f, "%d %d\n", &cooRowIndex[i], &cooColIndex[i]);
	    		cooVal[i] = 0.00001;

	    	} else if (data_type == 'f'){ // float input
	        	fscanf(f, "%d %d %lg\n", &cooRowIndex[i], &cooColIndex[i], &cooVal[i]);
	        }
	        cooRowIndex[i]--;  
	        cooColIndex[i]--;

	        if (cooRowIndex[i] < 0 || cooColIndex[i] < 0) { // report error
	       		cout << "i = " << i << " [" <<cooRowIndex[i] << ", " << cooColIndex[i] << "] = " << cooVal[i] << endl;
	       	}
		}
	} else if(input_type == 'g') { // generate data
		//int n = 10000;
		n = atoi(filename);

		m = n;
		int nb = m / 8;
		double r;
		double r1 = 0.9;
		double r2 = 0.01;

		long long p = 0;

		for (int i = 0; i < m; i += nb) {
			if (i == 0) {
				r = r1;
			} else {
				r = r2;
			}
			for (int ii = i; ii < i + nb; ii++) {
				for (int j = 0; j < n * r; j++) {
					p++;
				}
			}
		}


		nnz = p;

		cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
		

		cudaMallocHost((void **)&cooRowIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooColIndex, nnz * sizeof(int));
	    cudaMallocHost((void **)&cooVal, nnz * sizeof(double));

	    p = 0;
		

		cout << "Start generating data " << std::flush;
		for (int i = 0; i < m; i += nb) {
			cout << "." << std::flush;
			if (i == 0) {
				r = r1;
			} else {
				r = r2;
			}
			//cout << "Matrix:" << endl;
			for (int ii = i; ii < i + nb; ii++) {
				for (int j = 0; j < n * r; j++) {
					//if (p > nnz) { cout << "error" << endl; break;}
					//else {

					cooRowIndex[p] = ii;
					cooColIndex[p] = j;
					cooVal[p] = (double) rand() / (RAND_MAX);
					p++;

					//cout << 1 << " ";
					//}
				}
				//cout << endl;
			}
		}
		cout << endl;

		//cout << "m: " << m << " n: " << n << " nnz: " << p << endl;


		cout << "Done generating data." << endl;


	}



    




	// Convert COO to CSR
    //csrRowPtr = (int *) malloc((m+1) * sizeof(int));
    cudaMallocHost((void **)&csrRowPtr, (m+1) * sizeof(long long));

    //cout << "m: " << m << " n: " << n << " nnz: " << nnz << endl;
    long long matrix_data_space = nnz * sizeof(double) + nnz * sizeof(int) + (m+1) * sizeof(int);
    //cout << matrix_data_space << endl;

	double matrix_size_in_gb = (double)matrix_data_space / 1e9;
    cout << "Matrix space size: " << matrix_size_in_gb << " GB." << endl;

    int * counter = new int[m];
    for (int i = 0; i < m; i++) {
    	counter[i] = 0;
    }
	for (int i = 0; i < nnz; i++) {
		counter[cooRowIndex[i]]++;
	}
	//cout << "nnz: " << nnz << endl;
	//cout << "counter: ";
	int t = 0;
	for (int i = 0; i < m; i++) {
		//cout << counter[i] << ", ";
		t += counter[i];
	}
	//cout << t << endl;
	//cout << endl;


	//cout << "csrRowPtr: ";
	csrRowPtr[0] = 0;
	for (int i = 1; i <= m; i++) {
		csrRowPtr[i] = csrRowPtr[i - 1] + counter[i - 1];
		//cout << "csrRowPtr[" << i <<"] = "<<csrRowPtr[i] << endl;
	}

	double * x;
	double * y1;
	double * y2;
	double * y3;

	//x = (double *)malloc(n * sizeof(double)); 
	//y1 = (double *)malloc(m * sizeof(double)); 
	y2 = (double *)malloc(m * sizeof(double)); 
	//y3 = (double *)malloc(m * sizeof(double)); 

	cudaMallocHost((void **)&x, n * sizeof(double));
	cudaMallocHost((void **)&y1, m * sizeof(double));
	//cudaMallocHost((void **)&y2, m * sizeof(double));
	cudaMallocHost((void **)&y3, m * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		x[i] = 1.0;//((double) rand() / (RAND_MAX)); 
	}


	for (int i = 0; i < m; i++)
	{
		y1[i] = 0.0;
		y2[i] = 0.0;
		y3[i] = 0.0;
	}

	double ALPHA = (double) rand() / (RAND_MAX);
	double BETA = (double) rand() / (RAND_MAX);

	double time_baseline = 0.0;
	double time_v1 = 0.0;
	double time_v2 = 0.0;

	double avg_time_baseline = 0.0;
	double avg_time_v1 = 0.0;
	double avg_time_v2 = 0.0;

	double curr_time = 0.0;

	int warm_up_iter = 1;

	double profile_time = 0.0;
	double min_profile_time = numeric_limits<double>::max();

	double best_dev_count = 0.0;
	double best_copy = 0.0;

	cout << "Warming up GPU(s)..." << endl;

	for (int i = 0; i < warm_up_iter; i++) {
		spMV_mgpu_v1(m, n, nnz, &ALPHA,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &BETA,
					 y2,
					 ngpu,
					 kernel_version);
	}

	
	for (int d = 1; d <= ngpu; d*=2) {
		for (int c = 1; c <= 8; c*=2) {
			curr_time = get_time();
			spMV_mgpu_v2(m, n, nnz, &ALPHA,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &BETA,
					 y3,
					 d,
					 kernel_version,
					 nnz / (d * c),
					 c);
			profile_time = get_time() - curr_time;	
			if (profile_time < min_profile_time) {
				min_profile_time = profile_time;
				best_dev_count = d;
				best_copy = c;
			}
		}
	}

	int ret1 = 0;
	int ret2 = 0;
	int ret3 = 0;
	
	cout << "Starting tests..." << endl;

	//cudaProfilerStart();

	cout << "  Test No.   Baseline    Version 1     Pass     Version 2     Pass" << endl;
	cout << "              Time(s)      Time(s)                Time(s)         " << endl;
	cout << "=======================================================================" << endl;

	for (int i = 0; i < repeat_test; i++) {
		for (int i = 0; i < m; i++)
		{
			y1[i] = 0.0;
			y2[i] = 0.0;
			y3[i] = 0.0;
		}

		curr_time = get_time();
		ret1 = spMV_mgpu_baseline(m, n, nnz, &ALPHA,
							 cooVal, csrRowPtr, cooColIndex, 
							 x, &BETA,
							 y1,
							 ngpu);
		time_baseline = get_time() - curr_time;	


		curr_time = get_time();
		ret2 = spMV_mgpu_v1(m, n, nnz, &ALPHA,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &BETA,
					 y2,
					 ngpu,
					 kernel_version);
		time_v1 = get_time() - curr_time;	
		
		//cudaProfilerStart();

		curr_time = get_time();
		ret3 = spMV_mgpu_v2(m, n, nnz, &ALPHA,
					 cooVal, csrRowPtr, cooColIndex, 
					 x, &BETA,
					 y3,
					 best_dev_count,
					 kernel_version,
					 nnz / (best_dev_count * best_copy),
					 best_copy);
		time_v2 = get_time() - curr_time;	

		
		avg_time_baseline += time_baseline;
		avg_time_v1  += time_v1;
		avg_time_v2  += time_v2;

		bool correct1 = true;
		bool correct2 = true;
		for(int i = 0; i < m; i++) {
			//cout << y1[i] << " - "  << y2[i] << " - "<< y3[i] << endl;
			if (abs(y1[i] - y2[i]) > 1e-3) {
				//cout << y1[i] << " - " << y3[i] << endl;
				correct1 = false;
			}
			if (abs(y1[i] - y3[i]) > 1e-3) {
				correct2 = false;
			}
		}

		cout << setw(10) << i+1;


		if (ret1 == 0) {
			cout << setw(11) << time_baseline;
		} else {
			cout << setw(11) << "Failed";
		}
		
		if (ret2 == 0) {
			cout << setw(13) << time_v1;
		} else {
			cout << setw(13) << "Failed";
		}

		if (ret1 == 0) {
			if (correct1) cout << setw(9) <<"Y";
			else cout << setw(9) << "N";
		} else {
			cout << setw(9) <<"N/A";
		}

		if (ret3 == 0) {
			cout << setw(14) << time_v2;
		} else {
			cout << setw(14) << "Failed.";
		}		

		if (ret1 == 0) {
			if (correct2) cout << setw(9) <<"Y";
			else cout << setw(9) << "N";
		} else {
			cout << setw(9) <<"N/A";
		}
		cout << endl;

	
	}

	//cudaProfilerStop();
	avg_time_baseline/=repeat_test;
	avg_time_v1/=repeat_test;
	avg_time_v2/=repeat_test;

	cout << "......................................................................." << endl;

	cout << setw(10) << "Average" << " ";
	if (ret1 == 0) {
		cout << setw(11) << avg_time_baseline;
	} else {
		cout << setw(11) << "Failed";
	}
	if (ret2 == 0) {
		cout << setw(13) << avg_time_v1;
	} else {
		cout << setw(13) << "Failed";
	}
	if (ret3 == 0) {
		cout << setw(23) << avg_time_v2;
	} else {
		cout << setw(23) << "Failed";
	}
	cout << endl;

	cudaFreeHost(cooRowIndex);
	cudaFreeHost(cooColIndex);
	cudaFreeHost(cooVal);
	cudaFreeHost(csrRowPtr);
	
}
