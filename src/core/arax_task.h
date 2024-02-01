#ifndef ARAX_TASK_HEADER
#define ARAX_TASK_HEADER
#include "core/arax_proc.h"
#include "async.h"

/**
 * Arax Task message.
 */
typedef struct arax_task_msg
{
    arax_object_s      obj;
    arax_pipe_s *      pipe;
    arax_accel *       accel;     /**< Accelerator responsible for this task */
    arax_proc *        proc;      /**< Process id */
    size_t             host_size; /**< Size of \c host_data in bytes */
    int                in_count;  /**< Number of input buffers */
    int                out_count; /**< Number of output buffers */
    arax_accel_type_e  type;      /**< Type of task at issue */
    arax_task_state_e  state;     /**< Current state of task. */
    arax_task_stats_s  stats;     /**< Task related statistics */
    async_completion_s done;      /**< Used for arax_task_mark_done(), arax_task_wait_done() */
    arax_data *        io[];      /**< Array of input and output buffers has in_count+out_count elements. The first in_count elements point to the inputs and the remaining out_count elements point to the outputs */
} arax_task_msg_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Allocates a \c arax_task_msg_s object.
 *
 * \param vpipe A valid \c arax_pipe_s instance.
 * \param accel \c arax_accel instance
 * \param pric \c arax_proc instance
 * \param host_size bytes to reserve for the tasks host_data.
 * \param ins number of inputs
 * @param dev_in array of arax_data pointers with input data.
 * \param outs number of outputs
 * @param dev_out array of arax_data pointers with output data.
 */
arax_task_msg_s* arax_task_alloc(arax_pipe_s *vpipe, arax_accel *accel, arax_proc *proc, size_t host_size, int ins,
  arax_data **dev_in, int outs, arax_data **dev_out);

/**
 * Returns start of host data of \c task.
 *
 * \param task A valid arax_task_msg_s
 * \param size Size of the host_data, this has to match the \c host_size given to \c arax_task_alloc()/\c arax_task_issue()
 * \return Pointer to host data if \c host_size > 0, null otherwise.
 */
void* arax_task_host_data(arax_task_msg_s *task, size_t size);

void arax_task_submit(arax_task_msg_s *task);

void arax_task_wait_done(arax_task_msg_s *msg);

void arax_task_mark_done(arax_task_msg_s *msg, arax_task_state_e state);
#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef ARAX_TASK_HEADER
