// #include "AraxLibUtilsGPU.h"
#include "AraxLibUtilsOpenCL.h"
#include "arax.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include <exception>
// #include "OpenclaccelThread.h"

#define BREAKDOWNS_VERBOSE
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION  120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#include "err_code.h"
#include <CL/cl.h>
#include <CLHelper.h>

// #define BREAKDOWNS_VERBOSE //Enable to distinguish time for malloc and memcpy
using namespace std;

extern cl_platform_id defaultPlatform;
extern cl_device_id defaultDevice;
extern cl_context defaultContext;

/********         PROC LIST FUNCTIONs            ********/
int alloc_opencl(arax_data_s *data);

/*
 * Checks if \c data has been allocated in FOGA, if not allocates.
 * \return True if already allocated or just allocated. False if allocation
 * failed.
 */
bool alloc_no_throttle(arax_data_s *data)
{
    size_t sz = arax_data_size(data);

    // data already allocated
    if (data->remote)
        return true;

    arax_assert(data->accel);
    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);
    data->phys = ((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);
    arax_accel_size_dec(((arax_vaccel_s *) (data->accel))->phys,
      arax_data_size(data));
    bool err = _clMallocRW(defaultContext, sz, data);

    #ifdef DEBUG
    if (err == false)
        return false;

    #endif
    // isAligned(data);
    return true;
}

ARAX_HANDLER_EX(alloc_data, OPEN_CL, cl_command_queue * stream){
    arax_assert(task->in_count == 0);
    arax_assert(task->out_count == 1);
    arax_data_s *data = (arax_data_s *) task->io[0];
    size_t sz         = arax_data_size(data);

    // data have not been allocated in the accelerator
    if (data->remote) {
        arax_task_mark_done(task, task_completed);
        return task_completed;
    }

    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);

    bool err = _clMallocRW(defaultContext, sz, data);

    #ifdef DEBUG
    if (err == false) {
        arax_task_mark_done(task, task_failed);
        return task_failed;
    }
    #endif
    arax_assert(data->remote);
    data->phys = ((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);

    arax_task_mark_done(task, task_completed);
    return task_completed;
}

/**
 * Performs a Remote to Remote copy
 */
ARAX_HANDLER_EX(memcpy, OPEN_CL, cl_command_queue * stream){
    memcpyArgs *args =
      (memcpyArgs *) arax_task_host_data(task, sizeof(memcpyArgs));
    arax_data_s *src = (arax_data_s *) (task->io[0]);
    arax_data_s *dst = (arax_data_s *) (task->io[1]);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " src: " << src
              << " offset: " << args->src_offset << " dst: " << dst
              << " offset: " << args->dst_offset << std::endl;
    #endif

    size_t sz = args->size;
    if (args->size == 0) {
        std::cerr << __FILE__ << " " << __func__
                  << " args->size is Zero. Please specify size! Abort.\n";
        abort();
    }

    size_t sz_src = arax_data_size(src);
    size_t sz_dst = arax_data_size(dst);

    if (!alloc_no_throttle(dst)) {
        arax_task_mark_done(task, task_failed);
        return task_failed;
    }
    if (!alloc_no_throttle(src)) {
        arax_task_mark_done(task, task_failed);
        return task_failed;
    }

    arax_assert(dst->remote);
    arax_assert(src->remote);

    if ((sz_dst - (args->dst_offset) < sz)) {
        cerr << __FILE__ << " " << __LINE__ << " " << __func__ << endl;
        cerr << "arax_data_size(dst)-dst_offset: " << sz_dst - args->dst_offset
             << " is < args->size: " << sz << endl;
        abort();
    }
    if ((sz_src - (args->src_offset)) < sz) {
        cerr << __FILE__ << " " << __LINE__ << " " << __func__ << endl;
        cerr << " arax_data_size(src)-src_offset: " << sz_src - (args->src_offset)
             << " is < args->size: " << sz << endl;
        abort();
    }
    _clMemcpyD2D(*stream, src, dst, sz_src, args->src_offset, args->dst_offset);

    arax_task_mark_done(task, task_completed);
    // If arax_task is async free task from controller
    if (args->sync == false) {
        _clFinish(*stream);
        arax_task_free(task);
    }
    return task_completed;
} // opencl_memcpy

ARAX_HANDLER_EX(memset, OPEN_CL, cl_command_queue * stream){
    memsetArgs *args =
      (memsetArgs *) arax_task_host_data(task, sizeof(memsetArgs));
    arax_data_s *data = (arax_data_s *) (task->io[0]);

    // data have not been allocated in the accelerator
    if (!alloc_no_throttle(data)) {
        arax_task_mark_done(task, task_failed);
        return task_failed;
    }
    arax_assert(data->remote);
    // We do not need it opencl supports buffer base ptr and offset
    // char *dst = ((char *)data->remote) + args->data_offset;

    _clMemset(*stream, data, &args->value, sizeof(args->value), args->data_offset,
      arax_data_size(data));

    arax_task_mark_done(task, task_completed);
    arax_task_free(task);

    return task_completed;
}

ARAX_HANDLER_EX(free, OPEN_CL, cl_command_queue * stream){
    void **args              = (void **) arax_task_host_data(task, sizeof(void *) * 4);
    void *ptrAtDevice        = args[1];
    size_t size              = (size_t) args[2];
    arax_vaccel_s *accel_ptr = (arax_vaccel_s *) args[3];

    _clFree(ptrAtDevice);

    arax_accel_size_inc(accel_ptr, size);
    arax_task_mark_done(task, task_completed);
    arax_task_free(task);

    return task_completed;
}

ARAX_HANDLER_EX(init_phys, OPEN_CL, cl_command_queue * stream){
    _clFinish(*stream);
    arax_task_mark_done(task, task_completed);
    return task_completed;
}

ARAX_HANDLER_EX(arax_data_set, OPEN_CL, cl_command_queue * stream){
    arax_assert(task->in_count == 0);
    arax_assert(task->out_count == 1);
    void *host_src    = arax_task_host_data(task, task->host_size);
    arax_data_s *data = (arax_data_s *) (task->io[0]);

    if (!alloc_no_throttle(data)) {
        arax_task_mark_done(task, task_failed);
        return task_failed;
    }
    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data
              << " remote: " << data->remote << " data: " << data
              << " size: " << task->host_size << " host: " << host_src
              << " arax_data_size: " << arax_data_size(data) << std::endl;
    #endif
    arax_assert(data->remote);
    bool err = _clMemcpyH2D(*stream, data, task->host_size, host_src);
    _clFinish(*stream);
    CL_ERROR_FATAL(err);

    arax_task_free(task);
    return task_completed;
}

ARAX_HANDLER_EX(arax_data_get, OPEN_CL, cl_command_queue * stream){
    arax_assert(task->in_count == 0);
    arax_assert(task->out_count == 1);
    void *host_src    = arax_task_host_data(task, task->host_size);
    arax_data_s *data = (arax_data_s *) (task->io[0]);

    arax_assert(data->remote);
    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data
              << " remote: " << data->remote << std::endl;
    #endif
    bool err = _clMemcpyD2H(*stream, data, arax_data_size(data), host_src);

    CL_ERROR_FATAL(err);
    _clFinish(*stream);
    arax_task_mark_done(task, task_completed);

    return task_completed;
}
