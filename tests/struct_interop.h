#ifndef TEST_STRUCT_INTEROP_HEADER
#define TEST_STRUCT_INTEROP_HEADER
#include <stddef.h>
#include "vine_pipe.h"
#include "core/vine_data.h"
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
void get_c_sizes(size_t *sizes);

#ifndef __cplusplus
void get_cpp_sizes(size_t *sizes);
#endif

#ifdef __cplusplus
extern "C" void get_cpp_sizes(size_t *sizes)
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
    sizes[7]  = sizeof(vine_throttle_s);
    sizes[8]  = sizeof(vine_proc_s);
    sizes[9]  = sizeof(vine_accel_s);
    sizes[10] = sizeof(vine_data_s);
    sizes[11] = sizeof(vine_task_stats_s);
    sizes[12] = sizeof(vine_task_msg_s);
    sizes[13] = sizeof(vine_pipe_s);
    sizes[14] = sizeof(vine_vaccel_s);
}

#endif // ifndef TEST_STRUCT_INTEROP_HEADER
