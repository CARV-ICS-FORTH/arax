#ifndef TEST_STRUCT_INTEROP_HEADER
#define TEST_STRUCT_INTEROP_HEADER
#include <stddef.h>
#include "arax_pipe.h"
#include "core/arax_data.h"
#define STRUCT_INTEROP_SIZES 15

/*
 * Both functions put the same sizeof(struct) results in \c sizes.
 * Both functions have the same body, but are compilled with gcc and
 * g++ respectively.
 *
 * Due to diferences between c and c++, the same struct definition might have
 * different layout, and thus size.
 *
 * Thus comparing \c sizes, will show if we have 'hit' such difference.
 */
#ifdef __cplusplus
extern "C" void get_c_sizes(size_t *sizes);
void get_cpp_sizes(size_t *sizes)
#else
void get_c_sizes(size_t * sizes)
#endif /* ifdef __cplusplus */
{
    sizes[0]  = sizeof(async_semaphore_s);
    sizes[1]  = sizeof(async_completion_s);
    sizes[2]  = sizeof(async_condition_s);
    sizes[3]  = sizeof(struct arch_alloc_s);
    sizes[4]  = sizeof(utils_list_s);
    sizes[5]  = sizeof(utils_queue_s);
    sizes[6]  = sizeof(utils_kv_s);
    sizes[7]  = sizeof(arax_throttle_s);
    sizes[8]  = sizeof(arax_proc_s);
    sizes[9]  = sizeof(arax_accel_s);
    sizes[10] = sizeof(arax_data_s);
    sizes[11] = sizeof(arax_task_stats_s);
    sizes[12] = sizeof(arax_task_msg_s);
    sizes[13] = sizeof(arax_pipe_s);
    sizes[14] = sizeof(arax_vaccel_s);
}

#endif // ifndef TEST_STRUCT_INTEROP_HEADER
