#ifndef ARAXLIB_HIP_GPU_UTILS_HEADER
#define ARAXLIB_HIP_GPU_UTILS_HEADER

#include "AraxLibMgr.h"
#include "AraxLibUtilsMemOps.h"
#include "arax_pipe.h"
#include "hip/hip_runtime.h"
#include <vector>

#ifdef LIBRARY_BUILD

extern bool hip_Host2GPU(arax_task_msg_s *arax_task, std::vector<void *> &ioHD);
extern bool hip_Host2GPUAsync(arax_task_msg_s *arax_task, vector<void *> &ioHD,
                              hipStream_t stream);

/* hip Memcpy from Device to host*/
extern bool hip_GPU2Host(arax_task_msg_s *arax_task, std::vector<void *> &ioDH);
extern bool hip_GPU2HostAsync(arax_task_msg_s *arax_task, vector<void *> &ioDH,
                              hipStream_t stream);

/* Free Device memory */
extern bool hip_GPUMemFree(std::vector<void *> &io);

/* Reset GPU */
extern bool hip_shouldResetGpu();

#endif

#endif
