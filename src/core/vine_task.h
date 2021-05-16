#ifndef VINE_TASK_HEADER
#define VINE_TASK_HEADER
#include "core/vine_proc.h"
#include "async.h"

/**
 * Vineyard Task message.
 */
typedef struct vine_task_msg
{
    vine_object_s      obj;
    vine_pipe_s *      pipe;
    vine_accel *       accel;       /**< Accelerator responsible for this task */
    vine_proc *        proc;        /**< Process id */
    size_t             scalar_size; /**< Size of \c scalars in bytes */
    int                in_count;    /**< Number of input buffers */
    int                out_count;   /**< Number of output buffers */
    vine_accel_type_e  type;        /**< Type of task at issue */
    vine_task_state_e  state;       /**< Current state of task. */
    vine_task_stats_s  stats;       /**< Task related statistics */
    async_completion_s done;        /**< Used for vine_task_mark_done(), vine_task_wait_done() */
    vine_data *        io[];        /**< Array of input and output buffers has in_count+out_count elements. The first in_count elements point to the inputs and the remaining out_count elements point to the outputs */
} vine_task_msg_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

vine_task_msg_s* vine_task_alloc(vine_pipe_s *vpipe, size_t scalar_size, int ins, int outs);

/**
 * Returns start of scalar array of \c task.
 *
 * \param task A valid vine_task_msg_s
 * \param size Size of the scalars, this has to match the \c scalar_size given to \c vine_task_alloc()/\c vine_task_issue()
 * \return Pointer to scalars if \c scalar_size > 0, null otherwise.
 */
void* vine_task_scalars(vine_task_msg_s *task, size_t size);

void vine_task_submit(vine_task_msg_s *task);

void vine_task_wait_done(vine_task_msg_s *msg);

void vine_task_mark_done(vine_task_msg_s *msg, vine_task_state_e state);
#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef VINE_TASK_HEADER
