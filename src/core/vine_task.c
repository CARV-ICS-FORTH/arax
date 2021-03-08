#include "vine_talk.h"
#include "vine_pipe.h"
#include "vine_data.h"
#include "utils/timer.h"
#include <stdlib.h>

vine_task_msg_s* vine_task_alloc(vine_pipe_s *vpipe, int ins, int outs)
{
    vine_task_msg_s *task;

    task = (vine_task_msg_s *) vine_object_register(&(vpipe->objs),
        VINE_TYPE_TASK, "Task",
        sizeof(vine_task_msg_s) + sizeof(vine_data *) * (ins + outs), 1);

    if (!task)     // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    async_completion_init(&(vpipe->async), &(task->done));

    task->pipe      = vpipe;
    task->in_count  = ins;
    task->out_count = outs;
    task->args      = 0;

    return task;
}

void vine_task_submit(vine_task_msg_s *task)
{
    utils_queue_s *queue = 0;

    vine_object_s *accel = task->accel;

    switch (accel->type) {
        case VINE_TYPE_PHYS_ACCEL: {
            task->type = ((vine_accel_s *) accel)->type;
            queue      = task->pipe->queue;
            break;
        }
        case VINE_TYPE_VIRT_ACCEL: {
            task->type = ((vine_vaccel_s *) accel)->type;
            queue      = vine_vaccel_queue((vine_vaccel_s *) accel);
            break;
        }
        default:// GCOV_EXCL_START
        {
            fprintf(stderr, "Non accelerator type(%d) in %s!\n", accel->type, __func__);
            vine_assert(!"Expected vine_task_msg_s");
        } // GCOV_EXCL_STOP
    }

    vine_object_ref_inc(accel);

    utils_timer_set(task->stats.task_duration, start);
    /* Push it or spin */
    while (!utils_queue_push(queue, task) )
        ;
    task->state = task_issued;
    vine_pipe_add_task(task->pipe, task->type, ((vine_vaccel_s *) accel)->assignee);
}

void vine_task_wait_done(vine_task_msg_s *msg)
{
    async_completion_wait(&(msg->done));
}

void vine_task_mark_done(vine_task_msg_s *msg, vine_task_state_e state)
{
    msg->state = state;
    async_completion_complete(&(msg->done));
}

VINE_OBJ_DTOR_DECL(vine_task_msg_s)
{
    vine_task_msg_s *_task = (vine_task_msg_s *) obj;
    int cnt;

    // printf("\t%s %s %d\n","vine task DESTRUCTOR", ((vine_proc_s*)_task->proc)->obj.name, obj->ref_count);
    if (_task->args) {
        // printf("\t\tboom task args free %p with size:%lu\n",_task->args, VINE_DATA_ALLOC_SIZE(((vine_data_s*)_task->args)) );
        vine_object_ref_dec(_task->args);
    }


    for (cnt = 0; cnt < _task->in_count + _task->out_count; cnt++) {
        // printf("\t\tboom task data free %p\n",_task->io[cnt]);
        vine_object_ref_dec(_task->io[cnt]);
    }

    if (_task->accel)
        vine_object_ref_dec(_task->accel);
    else
        fprintf(stderr, "vine_task(%p,%s) dtor called, task possibly unissued!\n", obj, obj->name);

    arch_alloc_free(obj->repo->alloc, obj);
}
