#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <iostream>
#include <cmath>
#include <float.h>
//#include "anonymouslib_cuda.h"
#include "spmv_kernel.h"
#include <limits>

using namespace std;

int get_row_from_index(int n, long long * a, long long idx) {
	int l = 0;
	int r = n;
	while (l < r - 1 ) {
		//cout << "l = " << l <<endl;
		//cout << "r = " << r <<endl;
		int m = l + (r - l) / 2;
		//cout << "m = " << m <<endl;
		if (idx < a[m]) {
			r = m;
		} else if (idx > a[m]) {
			l = m;
		} else {
			return m;
		}
	}
	// cout << "a[" << l << "] = " <<  a[l];
	// cout << " a[" << r << "] = " << a[r];
	// cout << " idx = " << idx << endl;
	if (idx == a[l]) return l;
	if (idx == a[r]) return r;
	return l;

}

double get_time()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	double ms = (double)tp.tv_sec * 1000 + (double)tp.tv_usec / 1000; //get current timestamp in milliseconds
	//return 0.00001;
	return ms / 1000;
}


double get_gpu_availble_mem(int ngpu) {
	size_t uCurAvailMemoryInBytes;
	size_t uTotalMemoryInBytes;
	

	double min_mem = numeric_limits<double>::max();
	int device;
	for (device = 0; device < ngpu; ++device) 
	{
		cudaSetDevice(device);
		cudaMemGetInfo(&uCurAvailMemoryInBytes, &uTotalMemoryInBytes);
		//cout << uCurAvailMemoryInBytes << "/" << uTotalMemoryInBytes << endl;
		double aval_mem = (double)uCurAvailMemoryInBytes/1e9;
		//cout << aval_mem << endl;
		if (aval_mem < min_mem) {
			min_mem = aval_mem;
		}
	    // cudaDeviceProp deviceProp;
	    // cudaGetDeviceProperties(&deviceProp, device);
	    // printf("Device %d has compute capability %d.%d.\n",
	    //        device, deviceProp.major, deviceProp.minor);
	}


	return min_mem;
}
