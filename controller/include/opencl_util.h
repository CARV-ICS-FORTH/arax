#ifndef OPENCL_UTIL_H_
#define OPENCL_UTIL_H_

#include "CLHelper.h"
#include <CL/cl.h>
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STRING_BUFFER_LEN 1024
#define AOCL_ALIGNMENT 64

#define RED "\033[1;31m"
#define RESET "\033[0m"
#define ERROR_CHECKING

#ifdef ERROR_CHECKING
#define CL_ERROR_FATAL(err) clErrorCheckFatal(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
clErrorCheckFatal(bool err, const char *func, const char *file, size_t line) {
  if (err != true) {
    std::cerr << RED << func << " error : " << RESET << std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    arax_assert(!"Fatality");
  }
}
#else
#define CL_ERROR_FATAL(err)
#endif

// For functions that "return" the error code
#define CL_SAFE_CALL(...)                                                      \
  do {                                                                         \
    cl_int __ret = __VA_ARGS__;                                                \
    if (__ret != CL_SUCCESS) {                                                 \
      fprintf(stderr, "%s:%d: %s failed with error code ", __FILE__, __LINE__, \
              extractFunctionName(#__VA_ARGS__));                              \
      display_error_message(__ret, stderr);                                    \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
// Declaring some of the functions here to avoid reordering them
inline static char *extractFunctionName(const char *input);
inline static void display_error_message(cl_int errcode, FILE *out);

inline static void device_info_string(cl_device_id device, cl_device_info param,
                                      const char *name) {
  char string[STRING_BUFFER_LEN];
  CL_SAFE_CALL(
      clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &string, NULL));
  fprintf(stderr, "%-32s= %s\n", name, string);
}

inline static void device_info_device_type(cl_device_id device,
                                           cl_device_info param,
                                           const char *name) {
  cl_device_type device_type;
  CL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(cl_device_type),
                               &device_type, NULL));
  fprintf(stderr, "%-32s= %d\n", name, (int)device_type);
}

// Prints values that are ulong
inline static void device_info_ulong(cl_device_id device, cl_device_info param,
                                     const char *name) {
  cl_ulong size;
  CL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(cl_ulong), &size, NULL));
  if (param == CL_DEVICE_GLOBAL_MEM_SIZE) {
    fprintf(stderr, "%-32s= %0.3lf MBytes\n", name,
            (double)(size / (1024.0 * 1024.0)));
  } else if (param == CL_DEVICE_LOCAL_MEM_SIZE ||
             param == CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) {
    fprintf(stderr, "%-32s= %0.3lf KBytes\n", name, (double)(size / 1024.0));
  } else {
    fprintf(stderr, "%-32s= %lu\n", name, size);
  }
}

// Prints values that are ulong[3]
inline static void device_info_ulongarray(cl_device_id device,
                                          cl_device_info param,
                                          const char *name) {
  cl_ulong *size = (cl_ulong *)malloc(sizeof(cl_ulong) * 3);
  CL_SAFE_CALL(
      clGetDeviceInfo(device, param, sizeof(cl_ulong) * 3, size, NULL));
  fprintf(stderr, "%-32s= ", name);
  for (int i = 0; i < 2; i++) {
    fprintf(stderr, "%lu, ", size[i]);
  }
  fprintf(stderr, "%lu\n", size[2]);
}

// Displays available platforms and devices
inline static void display_device_info(cl_platform_id **platforms,
                                       cl_uint *platformCount) {
  unsigned i, j;
  cl_int error;
  cl_uint deviceCount;
  cl_device_id *devices;

  // Get all platforms
  CL_SAFE_CALL(clGetPlatformIDs(1, NULL, platformCount));
  *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * (*platformCount));
  CL_SAFE_CALL(clGetPlatformIDs((*platformCount), *platforms, NULL));

  fprintf(stderr, "\nQuerying devices for info:\n");

  for (i = 0; i < *platformCount; i++) {
    // Get all devices
    error = clGetDeviceIDs((*platforms)[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                           &deviceCount);
    if (error != CL_SUCCESS) {
      if (error == CL_DEVICE_NOT_FOUND) // No compatible OpenCL devices?
      {
        fprintf(stderr, "======================================================"
                        "==========================\n");
        fprintf(stderr, "Platform number %d:\n\n", i);
        fprintf(stderr, "No devices were found in this platform!\n");
        fprintf(stderr, "======================================================"
                        "==========================\n\n");
      } else {
        fprintf(stderr, "%s:%d: clGetDeviceIDs() failed with error code ",
                __FILE__, __LINE__);
        display_error_message(error, stderr);
        exit(-1);
      }
    } else {
      devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
      CL_SAFE_CALL(clGetDeviceIDs((*platforms)[i], CL_DEVICE_TYPE_ALL,
                                  deviceCount, devices, NULL));

      for (j = 0; j < deviceCount; j++) {
        fprintf(stderr, "======================================================"
                        "==========================\n");
        fprintf(stderr,
                "Platform number %d, device number %d (device count: %d):\n\n",
                i, j, deviceCount);
        device_info_string(devices[j], CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
        device_info_string(devices[j], CL_DEVICE_NAME, "CL_DEVICE_NAME");
        device_info_string(devices[j], CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
        device_info_ulong(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                          "CL_DEVICE_GLOBAL_MEM_SIZE");
        device_info_ulong(devices[j], CL_DEVICE_LOCAL_MEM_SIZE,
                          "CL_DEVICE_LOCAL_MEM_SIZE");
        device_info_ulong(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                          "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
        device_info_device_type(devices[j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
        fprintf(stderr, "======================================================"
                        "==========================\n\n");
      }
    }
  }
}
inline static char *getVersionedKernelName(const char *kernel_name) {
  int slen = strlen(kernel_name) + 32;
  char *vname = (char *)malloc(sizeof(char) * (slen));
  // versioning
  snprintf(vname, slen, "%s.aocx", kernel_name);
  fprintf(stderr, "Using kernel file: %s\n", vname);
  return vname;
}

inline static char *read_kernel(const char *kernel_file_path,
                                size_t *source_size) {
  // Open kernel file
  FILE *kernel_file;
  kernel_file = fopen(kernel_file_path, "rb");
  if (!kernel_file) {
    fprintf(stderr, "Failed to open input kernel file \"%s\".\n",
            kernel_file_path);
    exit(-1);
  }

  // Determine the size of the input kernel or binary file
  fseek(kernel_file, 0, SEEK_END);
  *source_size = ftell(kernel_file);
  rewind(kernel_file);

  // Allocate memory for the input kernel or binary file
  char *source = (char *)calloc(*source_size + 1, sizeof(char));
  if (!source) {
    fprintf(stderr, "Failed to allocate enough memory for kernel file.\n");
    exit(-1);
  }

  // Read the input kernel or binary file into memory
  if (!fread(source, 1, *source_size, kernel_file)) {
    fprintf(stderr, "Failed to read kernel file into memory.\n");
    exit(-1);
  }
  fclose(kernel_file);

  source[*source_size] = '\0';
  return source;
}
// Validates device type selection and exports context properties
inline static void validate_selection(cl_platform_id *platforms,
                                      cl_uint *platformCount,
                                      cl_context_properties *ctxprop,
                                      cl_device_type *device_type) {
  unsigned i;
  cl_int error;
  cl_device_id *devices;
  cl_uint deviceCount;
  char deviceName[STRING_BUFFER_LEN];

  // Searching for compatible devices based on device_type
  // XXX MANOS was here : I modified i to start from 1 instead of 0 to ignore
  // Intel(R) FPGA Emulation Device
  for (i = 1; i < *platformCount; i++) {
    error = clGetDeviceIDs(platforms[i], *device_type, 0, NULL, &deviceCount);
    if (error != CL_SUCCESS) {
      if (error == CL_DEVICE_NOT_FOUND) // No compatible OpenCL devices?
      {
        fprintf(stderr, "======================================================"
                        "==========================\n");
        fprintf(stderr, "Platform number: %d\n", i);
        fprintf(
            stderr,
            "No compatible devices found, moving to next platform, if any.\n");
        fprintf(stderr, "======================================================"
                        "==========================\n\n");
      } else {
        fprintf(stderr, "%s:%d: clGetDeviceIDs() failed with error code ",
                __FILE__, __LINE__);
        display_error_message(error, stderr);
        exit(-1);
      }
    } else {
      devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
      CL_SAFE_CALL(clGetDeviceIDs(platforms[i], *device_type, deviceCount,
                                  devices, NULL));
      CL_SAFE_CALL(clGetDeviceInfo(devices[0], CL_DEVICE_NAME,
                                   STRING_BUFFER_LEN, &deviceName, NULL));

      fprintf(stderr, "========================================================"
                      "========================\n");
      fprintf(stderr, "Selected platform number: %d\n", i);
      fprintf(stderr, "Device count: %d\n", deviceCount);
      fprintf(stderr, "Device type: %d\n", (int)*device_type);
      fprintf(stderr, "Selected device: %s\n", deviceName);
      fprintf(stderr, "========================================================"
                      "========================\n\n");

      ctxprop[0] = CL_CONTEXT_PLATFORM;
      ctxprop[1] = (cl_context_properties)platforms[i];
      ctxprop[2] = 0;
      break;
    }
  }
}

inline static void display_error_message(cl_int errcode, FILE *out) {
  switch (errcode) {

  // Common error codes
  case CL_SUCCESS:
    fprintf(out, "CL_SUCCESS.\n");
    break;
  case CL_DEVICE_NOT_FOUND:
    fprintf(out, "CL_DEVICE_NOT_FOUND.\n");
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    fprintf(out, "CL_DEVICE_NOT_AVAILABLE.\n");
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    fprintf(out, "CL_COMPILER_NOT_AVAILABLE.\n");
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    fprintf(out, "CL_MEM_OBJECT_ALLOCATION_FAILURE.\n");
    break;
  case CL_OUT_OF_RESOURCES:
    fprintf(out, "CL_OUT_OF_RESOURCES.\n");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(out, "CL_OUT_OF_HOST_MEMORY.\n");
    break;
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    fprintf(out, "CL_PROFILING_INFO_NOT_AVAILABLE.\n");
    break;
  case CL_MEM_COPY_OVERLAP:
    fprintf(out, "CL_MEM_COPY_OVERLAP.\n");
    break;
  case CL_IMAGE_FORMAT_MISMATCH:
    fprintf(out, "CL_IMAGE_FORMAT_MISMATCH.\n");
    break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    fprintf(out, "CL_IMAGE_FORMAT_NOT_SUPPORTED.\n");
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    fprintf(out, "CL_BUILD_PROGRAM_FAILURE.\n");
    break;
  case CL_MAP_FAILURE:
    fprintf(out, "CL_MAP_FAILURE.\n");
    break;
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    fprintf(out, "CL_MISALIGNED_SUB_BUFFER_OFFSET.\n");
    break;
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    fprintf(out, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST.\n");
    break;
  case CL_INVALID_VALUE:
    fprintf(out, "CL_INVALID_VALUE.\n");
    break;
  case CL_INVALID_DEVICE_TYPE:
    fprintf(out, "CL_INVALID_DEVICE_TYPE.\n");
    break;
  case CL_INVALID_PLATFORM:
    fprintf(out, "CL_INVALID_PLATFORM.\n");
    break;
  case CL_INVALID_DEVICE:
    fprintf(out, "CL_INVALID_DEVICE.\n");
    break;
  case CL_INVALID_CONTEXT:
    fprintf(out, "CL_INVALID_CONTEXT.\n");
    break;
  case CL_INVALID_QUEUE_PROPERTIES:
    fprintf(out, "CL_INVALID_QUEUE_PROPERTIES.\n");
    break;
  case CL_INVALID_COMMAND_QUEUE:
    fprintf(out, "CL_INVALID_COMMAND_QUEUE.\n");
    break;
  case CL_INVALID_HOST_PTR:
    fprintf(out, "CL_INVALID_HOST_PTR.\n");
    break;
  case CL_INVALID_MEM_OBJECT:
    fprintf(out, "CL_INVALID_MEM_OBJECT.\n");
    break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    fprintf(out, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR.\n");
    break;
  case CL_INVALID_IMAGE_SIZE:
    fprintf(out, "CL_INVALID_IMAGE_SIZE.\n");
    break;
  case CL_INVALID_SAMPLER:
    fprintf(out, "CL_INVALID_SAMPLER.\n");
    break;
  case CL_INVALID_BINARY:
    fprintf(out, "CL_INVALID_BINARY.\n");
    break;
  case CL_INVALID_BUILD_OPTIONS:
    fprintf(out, "CL_INVALID_BUILD_OPTIONS.\n");
    break;
  case CL_INVALID_PROGRAM:
    fprintf(out, "CL_INVALID_PROGRAM.\n");
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    fprintf(out, "CL_INVALID_PROGRAM_EXECUTABLE.\n");
    break;
  case CL_INVALID_KERNEL_NAME:
    fprintf(out, "CL_INVALID_KERNEL_NAME.\n");
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    fprintf(out, "CL_INVALID_KERNEL_DEFINITION.\n");
    break;
  case CL_INVALID_KERNEL:
    fprintf(out, "CL_INVALID_KERNEL.\n");
    break;
  case CL_INVALID_ARG_INDEX:
    fprintf(out, "CL_INVALID_ARG_INDEX.\n");
    break;
  case CL_INVALID_ARG_VALUE:
    fprintf(out, "CL_INVALID_ARG_VALUE.\n");
    break;
  case CL_INVALID_ARG_SIZE:
    fprintf(out, "CL_INVALID_ARG_SIZE.\n");
    break;
  case CL_INVALID_KERNEL_ARGS:
    fprintf(out, "CL_INVALID_KERNEL_ARGS.\n");
    break;
  case CL_INVALID_WORK_DIMENSION:
    fprintf(out, "CL_INVALID_WORK_DIMENSION.\n");
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    fprintf(out, "CL_INVALID_WORK_GROUP_SIZE.\n");
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    fprintf(out, "CL_INVALID_WORK_ITEM_SIZE.\n");
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    fprintf(out, "CL_INVALID_GLOBAL_OFFSET.\n");
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    fprintf(out, "CL_INVALID_EVENT_WAIT_LIST.\n");
    break;
  case CL_INVALID_EVENT:
    fprintf(out, "CL_INVALID_EVENT.\n");
    break;
  case CL_INVALID_OPERATION:
    fprintf(out, "CL_INVALID_OPERATION.\n");
    break;
  case CL_INVALID_GL_OBJECT:
    fprintf(out, "CL_INVALID_GL_OBJECT.\n");
    break;
  case CL_INVALID_BUFFER_SIZE:
    fprintf(out, "CL_INVALID_BUFFER_SIZE.\n");
    break;
  case CL_INVALID_MIP_LEVEL:
    fprintf(out, "CL_INVALID_MIP_LEVEL.\n");
    break;
  case CL_INVALID_GLOBAL_WORK_SIZE:
    fprintf(out, "CL_INVALID_GLOBAL_WORK_SIZE.\n");
    break;
  default:
    fprintf(out, "Unknown OpenCL error code %d!\n", errcode);
    break;
  }
}

// Extract function name from __VA_ARGS__
inline static char *extractFunctionName(const char *input) {
  unsigned i;
  char *output = (char *)malloc(strlen(input) * sizeof(char));

  for (i = 0; i < strlen(input); i++) {
    output[i] = input[i];
    if (input[i] == '(') {
      break;
    }
  }
  output[i + 1] = ')';
  output[i + 2] = '\0';

  return output;
}

// Safe version of clBuildProgram() that automatically prints compilation log in
// case of failure
inline static void clBuildProgram_SAFE(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) {
  cl_int error = clBuildProgram(program, num_devices, device_list, options,
                                pfn_notify, user_data);

  if (error != CL_SUCCESS) {
    fprintf(stderr, "%s:%d: %s failed with error code ", __FILE__, __LINE__,
            "clBuildProgram()");
    display_error_message(error, stderr);

    if (error == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      char *log;

      // Get log size
      CL_SAFE_CALL(clGetProgramBuildInfo(
          program, *device_list, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));

      // Allocate memory for the log
      log = (char *)malloc(log_size);
      if (log == NULL) {
        fprintf(stderr, "Failed to allocate memory for compilation log");
        exit(-1);
      }

      // Get the log
      CL_SAFE_CALL(clGetProgramBuildInfo(
          program, *device_list, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));

      // Print the log
      fprintf(stderr, "\n=============================== Start of compilation "
                      "log ===============================\n");
      fprintf(stderr, "Build options: %s\n\n", options);
      fprintf(stderr, "%s", log);
      fprintf(stderr, "================================ End of compilation log "
                      "================================\n");
    }
    exit(-1);
  }
}

inline static void init_fpga(int *argc, char ***argv, int *version) {
  int shift = 0;
  int arg_idx = (*argc) - 1;
  fprintf(stderr, "Initialization\n");

  // Default version
  *version = 0;

  if (arg_idx > 0) {
    int ret = sscanf((*argv)[arg_idx], "v%d", version);
    if (ret == 1) {
      ++shift;
    }
  }

  // version number given
  fprintf(stderr, "Using verison %d\n", *version);

  // shift_argv(argc, argv, shift);
  *argc -= shift;
  return;
}

inline static void init_fpga2(int *argc, char ***argv, char **version_string,
                              int *version_number) {
  int shift = 0;
  int arg_idx = (*argc) - 1;
  fprintf(stderr, "Initialization\n");

  // Default version
  *version_number = 0;
  // default version string "v0"
  *version_string = (char *)malloc(sizeof(char) * 3);
  (*version_string)[0] = 'v';
  (*version_string)[1] = '0';
  (*version_string)[2] = '\0';

  if (arg_idx > 0) {
    int ret = sscanf((*argv)[arg_idx], "v%d", version_number);
    if (ret == 1) {
      ++shift;
      *version_string = (*argv)[arg_idx];
    }
  }

  // version number given
  fprintf(stderr, "Using version %d (%s)\n", *version_number, *version_string);

  // shift_argv(argc, argv, shift);
  *argc -= shift;
  return;
}

inline static void *alignedMalloc(size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, AOCL_ALIGNMENT, size)) {
    fprintf(stderr, "Aligned Malloc failed due to insufficient memory.\n");
    exit(-1);
  }
  return ptr;
}

inline static void *alignedCalloc(size_t num, size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, AOCL_ALIGNMENT, num * size)) {
    fprintf(stderr, "Aligned Calloc failed due to insufficient memory.\n");
    exit(-1);
  }
  memset(ptr, 0, size);
  return ptr;
}

#endif /* OPENCL_UTIL_H_ */
