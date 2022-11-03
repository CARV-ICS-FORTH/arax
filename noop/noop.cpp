#include "arax.h"
#include "core/arax_data.h"
#include <cstring>
#include <cstdlib>

#define MAGIC 1337

void noop_op(char *in, char *out, int l)
{
    int c;

    l -= 2;
    for (c = 0; l >= 0; l--, c++)
        out[c] = in[l];
    out[c] = 0;
}

#ifdef BUILD_MAIN
int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage:\n\t%s <string>\n\n", argv[0]);
        return 0;
    }
    arax_init();
    arax_accel *accel = arax_accel_acquire_type(CPU);
    arax_proc *proc   = arax_proc_get("noop");
    size_t size       = strlen(argv[1]) + 1;
    char *out         = (char *) calloc(size, 1);
    char *temp        = (char *) calloc(size, 1);
    arax_task *task;
    int magic = MAGIC;
    arax_buffer_s io[2] = {
        ARAX_BUFFER(size),
        ARAX_BUFFER(size)
    };

    arax_data_set(io[0], accel, argv[1]);

    task = arax_task_issue(accel, proc, &magic, 4, 1, io, 1, io + 1);

    arax_data_get(io[1], out);

    fprintf(stderr, "Noop is   \'%s\'\n", out);
    noop_op(argv[1], temp, size);
    fprintf(stderr, "Should be \'%s\'\n", temp);
    arax_data_free(io[0]);
    arax_data_free(io[1]);
    arax_task_free(task);
    arax_proc_put(proc);
    arax_accel_release(&accel);
    arax_exit();
    return strcmp(out, temp);
} // main

#endif // ifdef BUILD_MAIN

#include "core/arax_data_private.h"

#ifdef BUILD_SO
#include "core/arax_task.h"
#include <string>
#include <stdexcept>

ARAX_HANDLER(noop, CPU)
{
    int l     = arax_data_size(task->io[0]);
    char *in  = (char *) arax_data_deref(task->io[0]);
    char *out = (char *) arax_data_deref(task->io[1]);
    int magic = *(int *) arax_task_host_data(task, 4);

    if (magic != MAGIC) {
        throw std::runtime_error(std::string("Magic does not match ") + std::to_string(magic) + " != "
                + std::to_string(MAGIC));
    }
    noop_op(in, out, l);
    arax_task_mark_done(task, task_completed);
    return task_completed;
}

#endif // ifdef BUILD_SO
