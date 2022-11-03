#ifndef ARAXLIB_SDA_UTILS_HEADER
#define ARAXLIB_SDA_UTILS_HEADER

#include "AraxLibMgr.h"
#include "AraxLibUtilsMemOps.h"
#include "arax_pipe.h"
#include <vector>

#ifdef LIBRARY_BUILD
#include "xcl.h"
extern bool Host2SDA(arax_task_msg_s *arax_task, vector<void *> &ioHD,
                     xcl_world world, cl_kernel krnl);

extern bool SDA2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH,
                     xcl_world world, cl_kernel krnl);

extern bool SDAMemFree(vector<void *> &io);
#endif
#endif
