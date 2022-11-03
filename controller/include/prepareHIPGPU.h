#ifndef HIPACCEL_H
#define HIPACCEL_H

#include "hip/hip_runtime.h"

bool setHIPDevice(int id);
int gpuGetMaxGflopsDeviceId();
int numberOfHipDevices();
bool resetHIPDevice();
bool prepareHIPStream(hipStream_t str);
void hip_deviceSpecs(int number);
unsigned int hip_monitorGPU();
bool callInitKernel();
void call1Kernel();
#endif // ifndef HIPACCEL_H
