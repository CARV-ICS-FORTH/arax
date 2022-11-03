//------------------------------------------
//--cambine:helper function for OpenCL
//--programmer:	Jianbin Fang
//--date:	27/12/2010
//------------------------------------------
#ifndef _CL_HELPER_
#define _CL_HELPER_

#include "arax.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "definesEnable.h"
#include "opencl_util.h"
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;

#define CL_ERR_TO_STR(err)                                                     \
  case err:                                                                    \
    return #err
#define WORK_DIM 2 // work-items dimensions
char const *clGetErrorString(cl_int const err);
cl_int cl_status;

// Create read only buffer
// cl_mem _clMallocRW(cl_context context, int size, void *h_mem_ptr) {
bool _clMallocRW(cl_context context, int size, arax_data_s *data) {
  cl_mem d_mem =
      clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &cl_status);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << std::endl;
    return false;
  }
#endif
  data->remote = d_mem;
  return true;
}

// Transfer data from host to device
bool _clMemcpyH2D(cl_command_queue queue, arax_data_s *data, int size,
                  const void *h_mem_ptr) {
  cl_status = clEnqueueWriteBuffer(queue, (cl_mem)data->remote, CL_FALSE, 0,
                                   size, h_mem_ptr, 0, NULL, NULL);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << " line: " << __LINE__ << std::endl;
    return false;
  }
#endif
  return true;
}
//--------------------------------------------------------
// transfer data from device to host
bool _clMemcpyD2H(cl_command_queue queue, arax_data_s *data, int size,
                  void *h_mem) {
  cl_status = clEnqueueReadBuffer(queue, (cl_mem)data->remote, CL_FALSE, 0,
                                  size, h_mem, 0, 0, 0);

#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << " line: " << __LINE__ << std::endl;
    return false;
  }
#endif
  return true;
}
//--------------------------------------------------------
// Copy data device to device
bool _clMemcpyD2D(cl_command_queue queue, arax_data_s *src, arax_data_s *dst,
                  int size, int src_offset, int dst_offset) {
  cl_status =
      clEnqueueCopyBuffer(queue, (cl_mem)src->remote, (cl_mem)dst->remote,
                          src_offset, dst_offset, size, 0, NULL, NULL);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << " line: " << __LINE__ << std::endl;
    return false;
  }
#endif
  return true;
}
// Memset data device
bool _clMemset(cl_command_queue queue, arax_data_s *d_mem, void *pattern,
               size_t pattern_size, int src_offset, int size) {
  cl_status =
      clEnqueueFillBuffer(queue, (cl_mem)d_mem->remote, pattern, pattern_size,
                          src_offset, size, 0, NULL, NULL);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << std::endl;
    return false;
  }
#endif
  return true;
}
void _clSetArgs(cl_kernel kernel, int arg_idx, void *d_mem, int size = 0) {
  if (!size)
    cl_int err = clSetKernelArg(kernel, arg_idx, sizeof(d_mem), &d_mem);
  else
    cl_int err = clSetKernelArg(kernel, arg_idx, size, d_mem);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << std::endl;
    abort();
  }
#endif
}

bool _clFinish(cl_command_queue queue) {
  cl_status = clFinish(queue);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << std::endl;
    return false;
  }
#endif
  return true;
}
// release OpenCL objects
bool _clFree(void *ob) {
  if (ob != NULL)
    cl_status = clReleaseMemObject((cl_mem)ob);
#ifdef ERROR_CHECKING
  if (cl_status != CL_SUCCESS) {
    std::cerr << "Error in " << __func__ << " : " << clGetErrorString(cl_status)
              << std::endl;
    return false;
  }
#endif
  return true;
}

char const *clGetErrorString(cl_int const err) {
  switch (err) {
    CL_ERR_TO_STR(CL_SUCCESS);
    CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
    CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
    CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
    CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
    CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
    CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
    CL_ERR_TO_STR(CL_MAP_FAILURE);
    CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
    CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
    CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
    CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
    CL_ERR_TO_STR(CL_INVALID_VALUE);
    CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
    CL_ERR_TO_STR(CL_INVALID_PLATFORM);
    CL_ERR_TO_STR(CL_INVALID_DEVICE);
    CL_ERR_TO_STR(CL_INVALID_CONTEXT);
    CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
    CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
    CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
    CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
    CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
    CL_ERR_TO_STR(CL_INVALID_SAMPLER);
    CL_ERR_TO_STR(CL_INVALID_BINARY);
    CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
    CL_ERR_TO_STR(CL_INVALID_PROGRAM);
    CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
    CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
    CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
    CL_ERR_TO_STR(CL_INVALID_KERNEL);
    CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
    CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
    CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
    CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
    CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
    CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
    CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
    CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
    CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
    CL_ERR_TO_STR(CL_INVALID_EVENT);
    CL_ERR_TO_STR(CL_INVALID_OPERATION);
    CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
    CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
    CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
    CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
    CL_ERR_TO_STR(CL_INVALID_PROPERTY);
    CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
    CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
    CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
    CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);

  default:
    return "UNKNOWN ERROR CODE";
  }
}
#endif //_CL_HELPER_
