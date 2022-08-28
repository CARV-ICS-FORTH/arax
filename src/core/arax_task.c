#include "arax.h"
#include "arax_pipe.h"
#include "arax_data.h"
#include "utils/timer.h"
#include <stdlib.h>

arax_task_msg_s* arax_task_alloc(arax_pipe_s *vpipe, arax_accel *accel, arax_proc *proc, size_t host_size, int ins,
  arax_data **dev_in, int outs, arax_data **dev_out)
{
    arax_assert(accel);
    arax_assert(proc);
    // Size of io array
    const size_t io_size = sizeof(arax_data *) * (ins + outs);

    arax_task_msg_s *task;

    task = (arax_task_msg_s *) arax_object_register(&(vpipe->objs),
        ARAX_TYPE_TASK, "Task",
        sizeof(arax_task_msg_s) + io_size + host_size, 1);

    if (!task)     // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    async_completion_init(&(vpipe->async), &(task->done));

    task->accel     = accel;
    task->proc      = proc;
    task->pipe      = vpipe;
    task->in_count  = ins;
    task->out_count = outs;
    task->host_size = host_size;

    arax_data **dest = task->io;
    int cnt;

    for (cnt = 0; cnt < ins; cnt++, dest++) {
        *dest = dev_in[cnt];
        arax_data_input_init(*dest, accel);
        arax_data_annotate(*dest, "%s:in[%d]", ((arax_proc_s *) proc)->obj.name, cnt);
    }

    for (cnt = 0; cnt < outs; cnt++, dest++) {
        *dest = dev_out[cnt];
        arax_data_output_init(*dest, accel);
        arax_data_annotate(*dest, "%s:out[%d]", ((arax_proc_s *) proc)->obj.name, cnt);
    }

    return task;
} /* arax_task_alloc */

void* arax_task_host_data(arax_task_msg_s *task, size_t size)
{
    arax_assert_obj(task, ARAX_TYPE_TASK);
    arax_assert(size == task->host_size);

    if (task->host_size == 0)
        return 0;

    const size_t io_size = sizeof(arax_data *) * (task->in_count + task->out_count);

    return (char *) (task + 1) + io_size;
}

void arax_task_submit(arax_task_msg_s *task)
{
    arax_object_s *accel = task->accel;

    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);

    arax_object_ref_inc(accel);

    utils_timer_set(task->stats.task_duration, start);
    task->state = task_issued;
    arax_vaccel_add_task((arax_vaccel_s *) accel, task);
}

void arax_task_wait_done(arax_task_msg_s *msg)
{
    arax_assert(msg->state == task_issued || msg->state == task_completed);
    async_completion_wait(&(msg->done));
}

void arax_task_mark_done(arax_task_msg_s *msg, arax_task_state_e state)
{
    msg->state = state;
    async_completion_complete(&(msg->done));
}

ARAX_OBJ_DTOR_DECL(arax_task_msg_s)
{
    arax_task_msg_s *_task = (arax_task_msg_s *) obj;
    int cnt;

    for (cnt = 0; cnt < _task->in_count + _task->out_count; cnt++) {
        // printf("\t\tboom task data free %p\n",_task->io[cnt]);
        arax_object_ref_dec(_task->io[cnt]);
    }

    if (_task->accel)
        arax_object_ref_dec(_task->accel);
    else
        fprintf(stderr, "arax_task(%p,%s) dtor called, task possibly unissued!\n", obj, obj->name);
}
