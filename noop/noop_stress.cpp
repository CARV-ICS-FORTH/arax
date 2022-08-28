#include "arax.h"
#include "core/arax_data.h"
#include <cstring>
#include <vector>
#include <thread>
#include <iostream>

#define MAGIC 1337

void vac_per_thread(arax_proc *proc, size_t ops)
{
    arax_accel *accel = arax_accel_acquire_type(CPU);

    while (ops--) {
        size_t size = strlen("Hello") + 1;
        char *out   = new char[size];
        char *temp  = new char[size];
        arax_task *task;
        int magic = MAGIC;
        arax_buffer_s io[2] = {
            ARAX_BUFFER(size),
            ARAX_BUFFER(size)
        };

        arax_data_set(io[0], accel, "Hello");

        task = arax_task_issue(accel, proc, &magic, 4, 1, io, 1, io + 1);

        arax_task_wait(task);

        arax_data_get(io[0], out);

        arax_data_free(io[0]);
        arax_data_free(io[1]);
        arax_task_free(task);
    }

    arax_accel_release(&accel);
}

int main(int argc, char *argv[])
{
    arax_init();

    arax_proc *proc = arax_proc_get("noop");

    std::vector<std::thread> threads;

    for (int c = 0; c < 10; c++)
        threads.emplace_back(vac_per_thread, proc, 1000);

    for (std::thread & thread : threads)
        thread.join();

    arax_proc_put(proc);
    arax_exit();
    return 0;
}
