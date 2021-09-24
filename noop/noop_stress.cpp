#include "vine_talk.h"
#include "core/vine_data.h"
#include <cstring>
#include <vector>
#include <thread>
#include <iostream>

#define MAGIC 1337

void vac_per_thread(vine_proc *proc, size_t ops)
{
    vine_accel *accel = vine_accel_acquire_type(CPU);

    while (ops--) {
        size_t size = strlen("Hello") + 1;
        char *out   = new char[size];
        char *temp  = new char[size];
        vine_task *task;
        int magic = MAGIC;
        vine_buffer_s io[2] = {
            VINE_BUFFER(size),
            VINE_BUFFER(size)
        };

        vine_data_set(io[0], accel, "Hello");

        task = vine_task_issue(accel, proc, &magic, 4, 1, io, 1, io + 1);

        vine_task_wait(task);

        vine_data_get(io[0], out);

        vine_data_free(io[0]);
        vine_data_free(io[1]);
        vine_task_free(task);
    }

    vine_accel_release(&accel);
}

int main(int argc, char *argv[])
{
    vine_talk_init();

    vine_proc *proc = vine_proc_get("noop");

    std::vector<std::thread> threads;

    for (int c = 0; c < 10; c++)
        threads.emplace_back(vac_per_thread, proc, 1000);

    for (std::thread & thread : threads)
        thread.join();

    vine_proc_put(proc);
    vine_talk_exit();
    return 0;
}
