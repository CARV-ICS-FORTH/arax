#include "FreeThread.h"
#include "arax_pipe.h"
#include "core/arax_accel.h"
#include <iostream>

FreeThread::FreeThread(arax_pipe_s *pipe, Config &conf)
    : run(true), conf(conf), std::thread(FreeThread::thread, this, pipe)
{
    std::cerr << "Create FreeThread." << std::endl;
}

FreeThread::~FreeThread(){ }

void FreeThread::terminate()
{
    std::cerr << "Release FreeThread." << std::endl;

    run = false;
}

void FreeThread::thread(FreeThread *ft, arax_pipe_s *pipe)
{
    #ifdef FREE_THREAD
    arax_task_msg_s *task = 0;
    do {
        // iterate a vector<AccelConfig *> which is a vector all accelerators
        for (auto accel : ft->conf.getAccelerators()) {
            arax_vaccel_s *vac2Free   = accel->arax_accel->free_vaq;
            utils_queue_s *queue2Free = arax_vaccel_queue(vac2Free);
            if (utils_queue_used_slots(queue2Free) != 0) {
                task = (arax_task_msg_s *) utils_queue_pop(queue2Free);
                if (task) {
                    char *taskName =
                      ((arax_object_s) (((arax_proc_s *) (task->proc))->obj)).name;
                    std::cerr << __func__ << " task " << taskName << std::endl;
                    arax_object_ref_dec(&(task->obj));
                }
            }
        }
    } while (ft->run);
    #endif // ifdef FREE_THREAD
}
