#ifndef OPENCL_ACCELTHREAD
#define OPENCL_ACCELTHREAD

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#define __CL_ENABLE_EXCEPTIONS

#include "CL/opencl.h"
#include "timers.h"
//#include "AOCLUtils/aocl_utils.h"
#include <CL/cl.h>
#include <atomic>
#include <map>
#include <mutex>
#include <pthread.h>

// Enable synchronous execution in OpenclaccelThread
#define Synch

class OpenCLaccelThread;

#include "accelThread.h"

struct CL_file {
  string file;
  bool isBinary;
};
struct oclHandleStruct {
  cl_context context;
  cl_device_id *devices;
  cl_program program;
  cl_int cl_status;
  std::string error_str;
  //  std::vector<cl_kernel> kernel;
};

class OpenCLaccelThread : public accelThread {
public:
  OpenCLaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
  ~OpenCLaccelThread();
  virtual void printOccupancy();
  virtual size_t getAvailableSize();
  virtual size_t getTotalSize();
  static std::map<string, cl_kernel> kernels;
  IMPLEMENTS_DEVICE_BASE_OPS();

private:
  int64_t pciId;
  vector<CL_file> kernel_files;
  vector<string> kernel_names;
  int numberOfPlatforms, numberOfDevices;
  cl_command_queue stream;
  bool loadKernels();
  bool initDevice(int id);
  bool getNumberOfDevices();
  bool prepareDevice();
};
#endif
