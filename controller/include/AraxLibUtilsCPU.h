#ifndef ARAXLIB_CPU__UTILS_HEADER
#define ARAXLIB_CPU__UTILS_HEADER

#include "AraxLibMgr.h"
#include "AraxLibUtilsMemOps.h"
#include "arax_pipe.h"
#include <vector>
#ifdef LIBRARY_BUILD
extern void Host2CPU(arax_task_msg_s *arax_task, std::vector<void *> &ioHD);

/* Cuda Memcpy from Device to host*/
extern void CPU2Host(arax_task_msg_s *arax_task, std::vector<void *> &ioDH);

/* Free Device memory */
extern void CPUMemFree(std::vector<void *> &io);
#endif
#endif // ifndef ARAXLIB_CPU__UTILS_HEADER
