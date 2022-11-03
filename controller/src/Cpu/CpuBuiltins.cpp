#include "AraxLibUtilsCPU.h"
#include "arax.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
// #define DEBUG_PRINTS
arax_task_state_e nop(arax_task_msg_s *arax_task)
{
    arax_task_mark_done(arax_task, task_completed);
    return task_completed;
}

arax_task_state_e alloc_cpu_data(arax_task_msg_s *arax_task)
{
    arax_data_s *data = (arax_data_s *) arax_task->io[0];

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " data: " << data
              << std::endl;
    #endif
    arax_assert((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);
    data->phys = ((arax_accel_s *) ((arax_vaccel_s *) (data->accel))->phys);
    arax_task_mark_done(arax_task, task_completed);
    return task_completed;
}

arax_task_state_e cpu_memset(arax_task_msg_s *arax_task)
{
    memsetArgs *args =
      (memsetArgs *) arax_task_host_data(arax_task, sizeof(memsetArgs));
    arax_data_s *data = (arax_data_s *) (arax_task->io[0]);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " data: " << data
              << " offset: " << args->data_offset
              << " deref: " << arax_data_deref(data) << std::endl;
    #endif

    memset((char *) arax_data_deref(data) + args->data_offset, args->value,
      args->size);
    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " ptr: " << data
              << " value: " << *((float *) arax_data_deref(data)) << std::endl;
    #endif
    arax_task_mark_done(arax_task, task_completed);
    // arax_task_free(arax_task);
    return task_completed;
}

arax_task_state_e cpu_memcpy(arax_task_msg_s *arax_task)
{
    memcpyArgs *args =
      (memcpyArgs *) arax_task_host_data(arax_task, sizeof(memcpyArgs));
    arax_data_s *src = (arax_data_s *) (arax_task->io[0]);
    arax_data_s *dst = (arax_data_s *) (arax_task->io[1]);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " src: " << src
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
    std::cerr << __func__ << " BEFORE task: " << arax_task << " dst: " << dst
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
    std::cerr << __func__ << " AFTER task: " << arax_task << " dst: " << dst
              << " dst value: "
              << *((char *) arax_data_deref(dst) + args->dst_offset)
              << " src: " << src << " src value: "
              << *((char *) arax_data_deref(src) + args->src_offset) << std::endl;
    #endif

    arax_task_mark_done(arax_task, task_completed);
    if (args->sync == false) {
        arax_task_free(arax_task);
        // arax_data_free(arax_task->io[0]);
    }
    return task_completed;
} // cpu_memcpy

arax_task_state_e cpu_memfree(arax_task_msg_s *arax_task)
{
    void **args       = (void **) arax_task_host_data(arax_task, sizeof(void *) * 4);
    arax_data_s *data = (arax_data_s *) args[0];

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " data: " << data
              << std::endl;
    #endif
    arax_accel_size_inc((arax_vaccel_s *) data->phys, arax_data_size(data));
    arax_task_mark_done(arax_task, task_completed);
    arax_task_free(arax_task);
    return task_completed;
}

arax_task_state_e arax_data_get_cpu(arax_task_msg_s *arax_task)
{
    arax_assert(arax_task->in_count == 0);
    arax_assert(arax_task->out_count == 1);
    arax_data_s *data = (arax_data_s *) (arax_task->io[0]);
    size_t size       = arax_data_size(data);
    void *ud = arax_task_host_data(arax_task, size);

    memcpy(ud, arax_data_deref(data), size);
    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " data: " << data
              << " data value: " << ((float *) ud)[0] << std::endl;
    for (int i = 0; i < size; i++)
        std::cerr << __func__ << " data: " << *((float *) ud + i) << std::endl;
    #endif
    arax_task_mark_done(arax_task, task_completed);
    return task_completed;
}

arax_task_state_e arax_data_set_cpu(arax_task_msg_s *arax_task)
{
    arax_assert(arax_task->in_count == 0);
    arax_assert(arax_task->out_count == 1);
    arax_data_s *data = (arax_data_s *) (arax_task->io[0]);
    size_t size       = arax_data_size(data);
    void *ud = arax_task_host_data(arax_task, size);

    #ifdef DEBUG_PRINTS
    std::cerr << __func__ << " task: " << arax_task << " data: " << data
              << " deref: " << arax_data_deref(data) << " host: " << ud
              << std::endl;
    #endif
    memcpy(arax_data_deref(data), ud, size);
    arax_task_free(arax_task);
    return task_completed;
}

arax_task_state_e init_phys(arax_task_msg_s *arax_task)
{
    arax_task_mark_done(arax_task, task_completed);
    return task_completed;
}

ARAX_PROC_LIST_START()
ARAX_PROCEDURE("alloc_data", CPU, (AraxFunctor *) alloc_cpu_data, 0)
ARAX_PROCEDURE("free", CPU, (AraxFunctor *) cpu_memfree, 0)
ARAX_PROCEDURE("memset", CPU, (AraxFunctor *) cpu_memset, sizeof(memsetArgs))
ARAX_PROCEDURE("memcpy", CPU, (AraxFunctor *) cpu_memcpy, sizeof(memcpyArgs))
ARAX_PROCEDURE("arax_data_set", CPU, (AraxFunctor *) arax_data_set_cpu, 0)
ARAX_PROCEDURE("arax_data_get", CPU, (AraxFunctor *) arax_data_get_cpu, 0)
ARAX_PROCEDURE("init_phys", CPU, (AraxFunctor *) init_phys, 0)
ARAX_PROC_LIST_END()
