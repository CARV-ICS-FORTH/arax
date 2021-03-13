#include "vine_talk.h"
#include "core/vine_data.h"
#include "VineLibUtilsCPU.h"
#include <cstring>

#define MAGIC 1337

void noop_op(char *in, char *out, int l)
{
    int c;

    l -= 2;
    for (c = 0; l >= 0; l--, c++)
        out[c] = in[l];
    out[c] = 0;
}

#ifdef BUILD_SO
vine_task_state_e noop(vine_task_msg_s *msg)
{
    int l     = vine_data_size(msg->io[0]);
    char *in  = (char *) vine_data_deref(msg->io[0]);
    char *out = (char *) vine_data_deref(msg->io[1]);
    int magic = *(int *) vine_task_scalars(msg, 4);

    if (magic != MAGIC) {
        throw std::runtime_error(std::string("Magic does not match ") + std::to_string(magic) + " != "
                + std::to_string(MAGIC));
    }
    noop_op(in, out, l);
    vine_data_modified(msg->io[1], SHM_SYNC);
    vine_task_mark_done(msg, task_completed);
    return task_completed;
}

VINE_PROC_LIST_START()
VINE_PROCEDURE("noop", CPU, noop, 0)
VINE_PROCEDURE("noop", GPU, noop, 0)
VINE_PROC_LIST_END()
#endif // ifdef BUILD_SO

#ifdef BUILD_MAIN
int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage:\n\t%s <string>\n\n", argv[0]);
        return 0;
    }
    vine_talk_init();
    vine_accel *accel = vine_accel_acquire_type(CPU);
    vine_proc *proc   = vine_proc_get(CPU, "noop");
    size_t size       = strlen(argv[1]) + 1;
    char *out         = (char *) calloc(size, 1);
    char *temp        = (char *) calloc(size, 1);
    vine_task *task;
    int magic = MAGIC;
    vine_buffer_s io[2] = {
        VINE_BUFFER(argv[1], size),
        VINE_BUFFER(out,     size)
    };

    vine_data_modified(io[0], USER_SYNC);

    vine_data_sync_to_remote(accel, io[0], 0);

    task = vine_task_issue(accel, proc, &magic, 4, 1, io, 1, io + 1);

    vine_task_wait(task);

    vine_data_sync_from_remote(accel, io[1], 1);

    fprintf(stderr, "Noop is   \'%s\'\n", out);
    noop_op(argv[1], temp, size);
    fprintf(stderr, "Should be \'%s\'\n", temp);
    vine_data_free(io[0]);
    vine_data_free(io[1]);
    vine_task_free(task);
    vine_proc_put(proc);
    vine_accel_release(&accel);
    vine_talk_exit();
    return strcmp(out, temp);
} // main

#endif // ifdef BUILD_MAIN
