#include "CPUaccelThread.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "definesEnable.h"
#include "utils/timer.h"
#include <iostream>

using namespace std;

CPUaccelThread::CPUaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf)
    : accelThread(v_pipe, conf)
{ }

size_t CPUaccelThread::getAvailableSize()
{
    return arax_throttle_get_available_size(&(v_pipe->throttle));
}

size_t CPUaccelThread::getTotalSize()
{
    return arax_throttle_get_total_size(&(v_pipe->throttle));
}

bool CPUaccelThread::checkpoint(arax_task_msg_s *arax_task)
{
    cerr << "NOT implemented!!" << endl;
    return false;
}

CPUaccelThread::~CPUaccelThread(){ }

/*initializes the CPU accelerator*/
bool CPUaccelThread::acceleratorInit(){ return true; }

/*Releases the CPU accelerator*/
void CPUaccelThread::acceleratorRelease(){ }

/**
 * Transfer Function Implementations
 */

/**
 * TODO: UGLY HACK TO AVOID API change PLZ FIXME
 */
extern arax_pipe_s *vpipe_s;

void Host2CPU(arax_task_msg_s *arax_task, vector<void *> &ioHD)
{
    int in;

    for (in = 0; in < arax_task->in_count; in++) {
        arax_data *araxdata = arax_task->io[in];
        ioHD.push_back(arax_data_deref(araxdata));
    }
    for (int out = in; out < arax_task->out_count + in; out++) {
        arax_data *araxdata = arax_task->io[out];
        ioHD.push_back(arax_data_deref(araxdata));
    }
}

/* Cuda Memcpy from Device to host*/
void CPU2Host(arax_task_msg_s *arax_task, vector<void *> &ioDH)
{
    arax_task_mark_done(arax_task, task_completed);
}

/* Free Device memory */
void CPUMemFree(vector<void *> &io){ }

/***************** MIGRATION functions ******************/
// #define DEBUG_PRINTS
bool CPUaccelThread::alloc_no_throttle(arax_data_s *data){ return true; }

void CPUaccelThread::alloc_remote(arax_data_s *vdata)
{
    #ifdef DEBUG_PRINTS
    std::cerr << "CPU : " << __func__ << " data: " << vdata << std::endl;
    #endif

    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
    vdata->phys = ((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
}

void CPUaccelThread::sync_to_remote(arax_data_s *vdata)
{
    #ifdef DEBUG_PRINTS
    std::cerr << "CPU : " << __func__ << " data: " << vdata << std::endl;
    #endif
}

void CPUaccelThread::sync_from_remote(arax_data_s *vdata)
{
    #ifdef DEBUG_PRINTS
    std::cerr << "CPU : " << __func__ << " data: " << vdata << std::endl;
    #endif
}

void CPUaccelThread::free_remote(arax_data_s *vdata)
{
    #ifdef DEBUG_PRINTS
    std::cerr << "CPU : " << __func__ << " data: " << vdata << std::endl;
    #endif

    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (vdata->accel))->phys);
    arax_assert(!"Unexpected free_remote CPU");
    vdata->phys = 0;
}

USES_DEFAULT_EXECUTE_HOST_CODE(CPUaccelThread)
USES_NOOP_RESET(CPUaccelThread)

REGISTER_ACCEL_THREAD(CPUaccelThread)
