#ifndef VINE_TASK_HEADER
#define VINE_TASK_HEADER
#include "core/vine_object.h"
#include "core/vine_proc.h"
#include "core/vine_buffer.h"

/**
 * Vineyard Task message.
 */
typedef struct vine_task_msg {
	vine_object_s     obj;
	vine_accel        *accel; /**< Accelerator responsible for this task */
	vine_proc         *proc; /**< Process id */
	vine_buffer_s     args; /**< Packed process arguments */
	int               in_count; /**< Number of input buffers */
	int               out_count; /**< Number of output buffers */
	vine_task_state_e state;
	vine_task_stats_s stats;
	vine_accel_type_e type;		/** Type of task at issue */
	utils_breakdown_instance_s breakdown;
	vine_buffer_s     io[]; /**< in_count+out_count pointers
	*                       to input and output
	* buffers*/
} vine_task_msg_s;

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

	vine_task_msg_s * vine_task_alloc(vine_pipe_s *vpipe,int ins,int outs);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif
