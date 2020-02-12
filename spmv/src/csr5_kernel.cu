#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
#include "anonymouslib_cuda.h"

int csr5_kernel(int m, int n, int nnz, double * alpha,
				  double * csrVal, int * csrRowPtr, int * csrColIndex, 
				  double * x, double * beta,
				  double * y) {
		int err = 0;
		anonymouslibHandle<int, unsigned int, double> A(m, n);
		err = A.inputCSR(
			            nnz, 
						csrRowPtr, 
						csrColIndex, 
						csrVal);
		//cout << "inputCSR err = " << err << endl;
		err = A.setX(x);
		//cout << "setX err = " << err << endl;
		A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
		A.warmup();
		err = A.asCSR5();
		//cout << "asCSR5 err = " << err << endl;
		err = A.spmv(*(alpha), y);
		return err;
}