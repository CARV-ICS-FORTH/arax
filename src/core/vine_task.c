#include "vine_task.h"
#include "vine_talk.h"
#include "vine_pipe.h"
#include "vine_data.h"
#include "utils/breakdown.h"
#include "utils/timer.h"
#include <stdlib.h>

vine_task_msg_s * vine_task_alloc(vine_pipe_s *vpipe,int ins,int outs)
{
	vine_task_msg_s * task;

	task = (vine_task_msg_s *)vine_object_register( &(vpipe->objs),
													VINE_TYPE_TASK,"Task",
				sizeof(vine_task_msg_s)+sizeof(vine_data*)*(ins+outs));

	if(!task)
		return 0;

	async_completion_init(&(vpipe->async),&(task->done));

	task->pipe = vpipe;
	task->in_count = ins;
	task->out_count = outs;
	task->args = 0;

	utils_breakdown_instance_init(&(task->breakdown));

	return task;
}

void vine_task_submit(vine_task_msg_s * task)
{
	utils_queue_s * queue;
	
	utils_breakdown_advance(&(task->breakdown),"Issue");
	vine_object_s * accel = task->accel;

        if(accel->type == VINE_TYPE_PHYS_ACCEL)
        {
		task->type = ((vine_accel_s*)accel)->type;
                queue = task->pipe->queue;
        }
        else
        {
		task->type = ((vine_vaccel_s*)accel)->type;
                queue = vine_vaccel_queue((vine_vaccel_s*)accel);
        }

	utils_timer_set(task->stats.task_duration,start);
        /* Push it or spin */
        while ( !utils_queue_push( queue,task ) )
                ;
        task->state = task_issued;
        vine_pipe_add_task(task->pipe,task->type,((vine_vaccel_s*)accel)->assignee);
}

void vine_task_wait_done(vine_task_msg_s * msg)
{
	async_completion_wait(&(msg->done));
}

void vine_task_mark_done(vine_task_msg_s * msg,vine_task_state_e state)
{
	msg->state = state;
	async_completion_complete(&(msg->done));
}

VINE_OBJ_DTOR_DECL(vine_task_msg_s)
{
	vine_task_msg_s *_task = (vine_task_msg_s *)obj;
//	int cnt;
	utils_breakdown_advance(&(_task->breakdown),"TaskFree");

	if(_task->args)
		vine_object_ref_dec(_task->args);

	if( (_task->in_count||_task->out_count) && ((vine_data_s*)(_task->io[0]))->remote )
	{
		_task->proc = vine_proc_get(((vine_proc_s*)(_task->proc))->type,"task_free");
		vine_object_ref_inc(&(_task->obj));
		vine_task_submit(_task);
	}
	else
		arch_alloc_free(obj->repo->alloc,obj);
	#ifdef BREAKS_ENABLE
	if(_task->breakdown.stats)
		utils_breakdown_end(&(_task->breakdown));
	#endif

}
