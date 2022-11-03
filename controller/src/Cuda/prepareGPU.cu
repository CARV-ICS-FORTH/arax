#include "stdio.h"
#include <sys/time.h>
#include <iostream>
#include "../include/prepareGPU.cuh"
#include "cuda_utils.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define RED     "\033[1;31m"
#define RESET   "\033[0m"

using namespace std;

__global__ void addkernel(int *data){
	*data += 1;
}
__global__ void addkernel2(){
	int i=0;
	int num=1;
	for (int j=0; j<10000; j++){
		i ++;
		num += gridDim.x * blockDim.x;
	}
}

/*Find the avaliable CUDA devices in the system*/
int numberOfCudaDevices() {
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	if (devCount > 0)
	{
		// Iterate through devices
		for (int i = 0; i < devCount; ++i)
		{
			cudaDeviceProp devProp;
			cudaGetDeviceProperties(&devProp, i);
			//deviceSpecs(devCount);
		}
		return devCount;
	} else {
		cout << "There is no CUDA device" << endl;
		return 0;
	}
}

/*Initiliazes CUDA devices*/
bool setCUDADevice(int id) {
	cudaError_t err;
	cout<<"Set Device with ID: "<< id <<endl;
	err = cudaSetDevice(id);
	if (err!= cudaSuccess)
	{
		cout<<"Failed to set Device with ID: "<< id <<" . ("<<cudaGetErrorString(err)<<")"<<endl;
		return false;
	}
	return true;

}

/*Performs a cudaMalloc and Free inorder to prepare the device*/
bool prepareCUDAStream(cudaStream_t stream) {
	struct timeval malloc_memcpy_st, malloc_memcpy_end;
	struct timeval krnl_st, krnl_end;
	struct timeval free_st, free_end;
	int *h_a, *d_a;
	cudaError_t err;

	gettimeofday(&malloc_memcpy_st,NULL);
	double t1 = malloc_memcpy_st.tv_sec  * 1000000 + malloc_memcpy_st.tv_usec;

	h_a = (int *)malloc(sizeof(int));
	err = cudaMalloc(&d_a, sizeof(int));
	
        CUDA_ERROR_FATAL(err);

	*h_a = 1;
	err = cudaMemcpyAsync(d_a, h_a, sizeof(int), cudaMemcpyHostToDevice, stream);

	CUDA_ERROR_FATAL(err);

	gettimeofday(&malloc_memcpy_end,NULL);
	double t2 = malloc_memcpy_end.tv_sec  * 1000000 +  malloc_memcpy_end.tv_usec;
	long double dur_malloc_memcpy = (t2 - t1)/1000 ;

	gettimeofday(&krnl_st,NULL);
	double kt1 = krnl_st.tv_sec  * 1000000 + krnl_st.tv_usec;

	addkernel<<<4,512,0,stream>>>(d_a);
	err = cudaGetLastError();
	
        CUDA_ERROR_FATAL(err);

	gettimeofday(&krnl_end,NULL);
	double kt2 = krnl_end.tv_sec  * 1000000 +  krnl_end.tv_usec;
	long double dur_krnl = (kt2 - kt1)/1000 ;

	gettimeofday(&free_st,NULL);
	double ft1 = free_st.tv_sec  * 1000000 + free_st.tv_usec;

	err = cudaFree(d_a);

	CUDA_ERROR_FATAL(err);

	gettimeofday(&free_end,NULL);
	double ft2 = free_end.tv_sec  * 1000000 +  free_end.tv_usec;
	long double dur_free = (ft2 - ft1)/1000 ;

	cerr<<"Malloc_Memcpy: "<<dur_malloc_memcpy<<" , krnl: "<< dur_krnl<<" , Free: "<<dur_free<<" ms"<<endl;
return true;
}

/*Reset CUDA devices*/
bool resetCUDADevice() {
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	if (cudaDeviceReset() != cudaSuccess) {
		return false;
	}
	return true;
}

unsigned int  monitorGPU()
{
	/*
	   nvmlInit();
	   int devicePCI ;
	   cudaGetDevice(&devicePCI);
	   nvmlDevice_t device;
	   unsigned int powerConsumption;

	   nvmlDeviceGetHandleByIndex(devicePCI, &device);
	   nvmlDeviceGetPowerUsage(device, &powerConsumption);
	   return powerConsumption;
	 */
	return 0;
}
/*GPUs specifications*/
void deviceSpecs(int deviceCount){
	int dev;
	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}
		printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
				deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
				deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
				deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
				deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxThreadsDim[0],
				deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Number of Concurrent kernels :       %d\n", deviceProp.concurrentKernels);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}
}
