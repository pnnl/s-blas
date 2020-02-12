#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "tranpose.h"
#include "sptrans_cuda.h"
#include "sptrans_kernal.h"

int main(int argc, char ** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");

    int m, n, nnzA, isSymmetricA;
    int *csrRowPtrA;
    int *csrColIdxA;
    double *csrValA;

    int *cscRowIdxA;
    int *cscColPtrA;
    double *cscValA;

    int device_id = 0;
    int dataformatted = DATAFORMAT_CSR;

    // "Usage: ``./sptrans -n (#gpu) -csr -mtx A.mtx'' 
    int argi = 1;

    // load number of GPU
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-n") != 0) return 0;
    
    int ngpu;
    if(argc > argi)
    {
        ngpu = atoi(argv[argi]);
        argi++;
    }

    int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount < ngpu) {
		printf("Error: Not enough number of GPUs. Only %i available\n", deviceCount);
		return -1;
	}
	if (ngpu <= 0) {
		printf("Error: Number of GPU(s) needs to be greater than 0.\n");
		return -1;
	}

   printf("Using %i GPU(s).\n", ngpu);


    // load format, csr or csc
    char *dataFormat;
    if(argc > argi)
    {
        dataFormat = argv[argi];
        argi++;
    }

    if (strcmp(dataFormat, "-csr") == 0)
        dataformatted = DATAFORMAT_CSR;
    else if (strcmp(dataFormat, "-csc") == 0)
        dataformatted = DATAFORMAT_CSC;
    printf("input data format = %s\n", dataFormat);
    printf("dataformatted = %i\n", dataformatted);

    // load matrix file type, mtx, cscl, or cscu
    char *matstr;
    if(argc > argi)
    {
        matstr = argv[argi];
        argi++;
    }
    printf("matstr = %s\n", matstr);

    // load matrix data from file
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    printf("-------------- %s --------------\n", filename);

    srand(time(NULL));
    if (strcmp(matstr, "-mtx") == 0)
    {
        // load mtx data to the csr format
        mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
        csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
        csrColIdxA = (int *)malloc(nnzA * sizeof(int));
        csrValA    = (double *)malloc(nnzA * sizeof(double));
        mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
        printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
/*
	printf("\n csrColIdx = [");
        for(int j = 0; j < nnzA; j++) printf(" %d ", csrColIdxA[j]);
	printf("]\n");
	
	printf("csrRowPtr =[");
	for(int j = 0; j < m+1; j++) printf(" %d ", csrRowPtrA[j]);
        printf("]\n");
*/
	int nnz_pointer = 0;

        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            {   
                        csrValA[nnz_pointer] = rand() % 10 + 1; //csrValA[j]; 
                        nnz_pointer++;
            }
 
        }

	cscRowIdxA = (int *)malloc(nnzA * sizeof(int));
        cscColPtrA = (int *)malloc((n+1) * sizeof(int));
        memset(cscColPtrA, 0, (n+1) * sizeof(int));
        cscValA    = (double *)malloc(nnzA * sizeof(double));

  struct timeval t1, t2;
    double time_cpu_trans= 0;
    gettimeofday(&t1, NULL);

        // transpose from csr to csc
        matrix_transposition(m, n, nnzA,
                             csrRowPtrA, csrColIdxA, csrValA,
                             cscRowIdxA, cscColPtrA, cscValA);

        
 gettimeofday(&t2, NULL);
    time_cpu_trans = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    
    printf("matrix trans used %4.8f ms in cpu node,\n",time_cpu_trans);
   
       

    }
    else{
	printf("donot support other format now, waiting for update!");
	
    }

            // test cpu result


    int *csrRowPtrB;
    int *csrColIdxB;
    double *csrValB;

    csrRowPtrB = (int *)malloc((m+1) * sizeof(int));
    csrColIdxB = (int *)malloc(nnzA * sizeof(int));
    memset(csrRowPtrB, 0, (m+1) * sizeof(int));
    csrValB    = (double *)malloc(nnzA * sizeof(double));


    //    // transpose from csc to csrB
    matrix_transposition_back(n, m, nnzA,
                         cscColPtrA, cscRowIdxA, cscValA,
                         csrColIdxB, csrRowPtrB, csrValB);

    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < nnzA; i++)
    {
        ref += abs(csrValA[i]);
        res += abs(csrValB[i] - csrValA[i]);
      // if (csrValA[i] != csrValB[i]) printf ("[%i, %d] csrValA = %f, csrValB = %f\n", i,  csrColIdxA[i], csrValA[i], csrValB[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("matrix transposition in cpu: passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("matrix transposition in cpu: _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    free(csrColIdxB);
    free(csrValB);
    free(csrRowPtrB);



     // set device
    //cudaSetDevice(device_id);
  //  cudaDeviceProp deviceProp;
   // cudaGetDeviceProperties(&deviceProp, device_id);
   //printf("---------------------------------------------------------------------------------------------\n");
   //printf("Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);



    // run cuda trans

    int *cscRowIdx;
    int *cscColPtr;
    double *cscVal;

    cscRowIdx = (int *)malloc(nnzA * sizeof(int));
    cscColPtr = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtr, 0, (n+1) * sizeof(int));
    cscVal    = (double *)malloc(nnzA * sizeof(double));


    printf("---------------------------------------------------------------------------------------------\n");
    
    cuda_sptrans(m, n, nnzA,
                 csrRowPtrA, csrColIdxA, csrValA,
                 cscRowIdx, cscColPtr, cscVal,
		 cscRowIdxA, cscColPtrA, cscValA);


    printf("---------------------------------------------------------------------------------------------\n");


    int *cscRowIdx2;
    int *cscColPtr2;
    double *cscVal2;

    cscRowIdx2 = (int *)malloc(nnzA * sizeof(int));
    cscColPtr2 = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtr2, 0, (n+1) * sizeof(int));
    cscVal2    = (double *)malloc(nnzA * sizeof(double));	

    printf("---------------------------------------------------------------------------------------------\n");
    
    kernal_sptrans(m, n, nnzA, ngpu,
                 csrRowPtrA, csrColIdxA, csrValA,
                 cscRowIdx2, cscColPtr2, cscVal2,
		 cscRowIdxA, cscColPtrA, cscValA);


    printf("---------------------------------------------------------------------------------------------\n");



    // done!
    free(cscRowIdx);
    free(cscColPtr);
    free(cscVal);
    free(cscRowIdx2);
    free(cscColPtr2);
    free(cscVal2);
    free(cscRowIdxA);
    free(cscColPtrA);
    free(cscValA);
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);
    


    
    
	
   

    return 0;
}

