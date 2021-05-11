#include "vine_talk.h"
#include "vine_pipe.h"
#include "vine_data.h"
#include "utils/timer.h"
#include <stdlib.h>

vine_task_msg_s* vine_task_alloc(vine_pipe_s *vpipe, vine_accel *accel, vine_proc *proc, size_t scalar_size, int ins,
  int outs)
{
    vine_assert(accel);
    vine_assert(proc);
    // Size of io array
    const size_t io_size = sizeof(vine_data *) * (ins + outs);

    vine_task_msg_s *task;

    task = (vine_task_msg_s *) vine_object_register(&(vpipe->objs),
        VINE_TYPE_TASK, "Task",
        sizeof(vine_task_msg_s) + io_size + scalar_size, 1);

    if (!task)     // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    async_completion_init(&(vpipe->async), &(task->done));

    task->accel       = accel;
    task->proc        = proc;
    task->pipe        = vpipe;
    task->in_count    = ins;
    task->out_count   = outs;
    task->scalar_size = scalar_size;

    return task;
}

void* vine_task_scalars(vine_task_msg_s *task, size_t size)
{
    vine_assert_obj(task, VINE_TYPE_TASK);
    vine_assert(size == task->scalar_size);

    if (task->scalar_size == 0)
        return 0;

    const size_t io_size = sizeof(vine_data *) * (task->in_count + task->out_count);

    return (char *) (task + 1) + io_size;
}

void vine_task_submit(vine_task_msg_s *task)
{
    vine_object_s *accel = task->accel;

    vine_assert_obj(accel, VINE_TYPE_VIRT_ACCEL);

    vine_object_ref_inc(accel);

    utils_timer_set(task->stats.task_duration, start);
    task->state = task_issued;
    vine_vaccel_add_task((vine_vaccel_s *) accel, task);
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

    for (cnt = 0; cnt < _task->in_count + _task->out_count; cnt++) {
        // printf("\t\tboom task data free %p\n",_task->io[cnt]);
        vine_object_ref_dec(_task->io[cnt]);
    }

    if (_task->accel)
        vine_object_ref_dec(_task->accel);
    else
        fprintf(stderr, "vine_task(%p,%s) dtor called, task possibly unissued!\n", obj, obj->name);
}
