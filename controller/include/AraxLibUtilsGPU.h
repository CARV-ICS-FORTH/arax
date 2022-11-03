#ifndef ARAXLIB_GPU_UTILS_HEADER
#define ARAXLIB_GPU_UTILS_HEADER

#include "AraxLibMgr.h"
#include "AraxLibUtilsMemOps.h"
#include "arax_pipe.h"
#include <vector>
#ifdef LIBRARY_BUILD

extern bool Host2GPU(arax_task_msg_s *arax_task, std::vector<void *> &ioHD);
extern bool Host2GPUAsync(arax_task_msg_s *arax_task, vector<void *> &ioHD,
  cudaStream_t stream);

/* Cuda Memcpy from Device to host*/
extern bool GPU2Host(arax_task_msg_s *arax_task, std::vector<void *> &ioDH);
extern bool GPU2HostAsync(arax_task_msg_s *arax_task, vector<void *> &ioDH,
  cudaStream_t stream);

/* Free Device memory */
extern bool GPUMemFree(std::vector<void *> &io);

/* Reset GPU */
extern bool shouldResetGpu();
#endif // ifdef LIBRARY_BUILD

#endif // ifndef ARAXLIB_GPU_UTILS_HEADER
