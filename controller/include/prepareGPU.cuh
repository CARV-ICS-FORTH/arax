#ifndef GPUACCEL_CUH
#define GPUACCEL_CUH

#include <cuda_runtime_api.h>

bool setCUDADevice(int id);
int gpuGetMaxGflopsDeviceId();
int numberOfCudaDevices();
bool resetCUDADevice();
bool prepareCUDAStream(cudaStream_t str);
void deviceSpecs(int number);
unsigned int  monitorGPU();
bool callInitKernel();                                                                              
void call1Kernel();
#endif
