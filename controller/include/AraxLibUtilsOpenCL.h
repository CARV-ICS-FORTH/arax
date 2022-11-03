#ifndef ARAXLIB_OPENCL_UTILS_HEADER
#define ARAXLIB_OPENCL_UTILS_HEADER

#include "AraxLibMgr.h"
#include "AraxLibUtilsMemOps.h"
#include "arax_pipe.h"
#include <CL/cl.h>
#include <vector>
#ifdef LIBRARY_BUILD

extern bool Host2OpenCL(arax_task_msg_s *arax_task, std::vector<void *> &ioHD);

extern bool OpenCL2Host(arax_task_msg_s *arax_task, std::vector<void *> &ioDH);

extern bool OpenCLMemFree(std::vector<void *> &io);

/* Get the kernel object from a kernel function name.*/
extern cl_kernel OpenCLGetKernel(std::string name, cl_command_queue *stream);

cl_platform_id getDefaultPlatform();
cl_device_id getDefaultDevice();
cl_context getDefaultContext();

#endif
#endif
