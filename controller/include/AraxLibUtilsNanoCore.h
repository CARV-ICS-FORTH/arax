#ifndef ARAXLIB_NANO_CORE_UTILS_HEADER
#define ARAXLIB_NANO_CORE_UTILS_HEADER

#include <vector>

using namespace ::std;

#include "AraxLibUtilsMemOps.h"
#include <AraxLibMgr.h>
#include <arax_pipe.h>

#ifdef LIBRARY_BUILD
/// @brief  Transfers data from host memory to NanoCore.
///
/// @param  arax_task   Arax task information.
/// @param  ioHD        Data to transfer.
///
/// @retval true    The transmission was successful.
/// @retval false   The transmission failed.
///
extern bool Host2NanoCore(arax_task_msg_s *arax_task, vector<void *> &ioHD);

/// @brief  Transfers data from NanoCore to host memory.
///
/// @param  arax_task   Arax task information.
/// @param  ioHD        Data to transfer.
///
/// @retval true    The transmission was successful.
/// @retval false   The transmission failed.
///
extern bool NanoCore2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH);

/// @brief  Releases memory allocated on NanoCore.
///
/// @param  io  Address of the memory space on NanoCore to free.
///
/// @retval true    Freeing was successful.
/// @retval false   Freeing failed.
///
extern bool NanoCoreMemFree(vector<void *> &io);
#endif // ifdef LIBRARY_BUILD
#endif // !defined(ARAXLIB_UTILS_HEADER)
