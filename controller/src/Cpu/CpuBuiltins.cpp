#include "AraxLibUtilsCPU.h"
#include "arax.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
// #define DEBUG_PRINTS
ARAX_HANDLER(alloc_data, CPU){
    arax_data_s *data = (arax_data_s *) task->io[0];

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data << std::endl;
    #endif
    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);
    data->phys = ((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);
    arax_task_mark_done(task, task_completed);
    return task_completed;
}

ARAX_HANDLER(memset, CPU){
    memsetArgs *args =
      (memsetArgs *) arax_task_host_data(task, sizeof(memsetArgs));
    arax_data_s *data = (arax_data_s *) (task->io[0]);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data
              << " offset: " << args->data_offset
              << " deref: " << arax_data_deref(data) << std::endl;
    #endif

    memset((char *) arax_data_deref(data) + args->data_offset, args->value,
      args->size);
    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " ptr: " << data
              << " value: " << *((float *) arax_data_deref(data)) << std::endl;
    #endif
    arax_task_mark_done(task, task_completed);
    // arax_task_free(task);
    return task_completed;
}
ARAX_HANDLER(memcpy, CPU){
    memcpyArgs *args =
      (memcpyArgs *) arax_task_host_data(task, sizeof(memcpyArgs));
    arax_data_s *src = (arax_data_s *) (task->io[0]);
    arax_data_s *dst = (arax_data_s *) (task->io[1]);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " src: " << src
              << " offset: " << args->src_offset
              << " deref src: " << arax_data_deref(src) << " dst: " << dst
              << " offset: " << args->dst_offset
              << " deref dst: " << arax_data_deref(dst) << std::endl;
    #endif
    size_t sz = args->size;
    if (args->size == 0) {
        std::cerr << __FILE__ << " " << __func__
                  << " args->size is Zero. Please specify size! Abort.\n";
        abort();
    }

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " BEFORE task: " << task << " dst: " << dst
              << " dst value: "
              << *((char *) arax_data_deref(dst) + args->dst_offset)
              << " src: " << src << " src value: "
              << *((char *) arax_data_deref(src) + args->src_offset)
              << " dst off: " << args->dst_offset
              << " src off: " << args->src_offset << std::endl;
    #endif
    memcpy((char *) arax_data_deref(dst) + args->dst_offset,
      (char *) arax_data_deref(src) + args->src_offset, sz);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " AFTER task: " << task << " dst: " << dst
              << " dst value: "
              << *((char *) arax_data_deref(dst) + args->dst_offset)
              << " src: " << src << " src value: "
              << *((char *) arax_data_deref(src) + args->src_offset) << std::endl;
    #endif

    arax_task_mark_done(task, task_completed);
    if (args->sync == false) {
        arax_task_free(task);
        // arax_data_free(arax_task->io[0]);
    }
    return task_completed;
}

ARAX_HANDLER(free, CPU){
    void **args       = (void **) arax_task_host_data(task, sizeof(void *) * 4);
    arax_data_s *data = (arax_data_s *) args[0];

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data << std::endl;
    #endif
    arax_accel_size_inc((arax_vaccel_s *) data->phys, arax_data_size(data));
    arax_task_mark_done(task, task_completed);
    arax_task_free(task);
    return task_completed;
}

ARAX_HANDLER(arax_data_get, CPU){
    arax_assert(task->in_count == 0);
    arax_assert(task->out_count == 1);
    arax_data_s *data = (arax_data_s *) (task->io[0]);
    size_t size       = arax_data_size(data);
    void *ud = arax_task_host_data(task, size);

    memcpy(ud, arax_data_deref(data), size);
    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data
              << " data value: " << ((float *) ud)[0] << std::endl;
    for (int i = 0; i < size; i++)
        std::cerr << __func__ << " data: " << *((float *) ud + i) << std::endl;
    #endif
    arax_task_mark_done(task, task_completed);
    return task_completed;
}

ARAX_HANDLER(arax_data_set, CPU){
    arax_assert(task->in_count == 0);
    arax_assert(task->out_count == 1);
    arax_data_s *data = (arax_data_s *) (task->io[0]);
    size_t size       = arax_data_size(data);
    void *ud = arax_task_host_data(task, size);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << task << " data: " << data
              << " deref: " << arax_data_deref(data) << " host: " << ud
              << std::endl;
    #endif
    memcpy(arax_data_deref(data), ud, size);
    arax_task_free(task);
    return task_completed;
}

ARAX_HANDLER(init_phys, CPU){
    arax_task_mark_done(task, task_completed);
    return task_completed;
}
