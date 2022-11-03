#include <iostream>
#include <mutex>

// #define DEBUG_NN
using namespace ::std;
#include "HeterogeneousScheduler.h"
HeterogeneousScheduler::HeterogeneousScheduler(picojson::object args)
    : Scheduler(args){ }

HeterogeneousScheduler::~HeterogeneousScheduler(){ }

HeterogeneousScheduler::State HeterogeneousScheduler::checkForRunnableTask(accelThread *th,
  arax_vaccel_s *                                                                       vaq)
{
    arax_task_msg_s *task = 0;
    arax_proc_s *proc     = 0;
    int runnable = -1;
    arax_accel_type_e accelThreadType = th->getAccelType();

    // Peek a task to check if it is runnable
    task = (arax_task_msg_s *) utils_queue_peek(arax_vaccel_queue(vaq));
    if (!task) {
        return NT;
    }

    char *taskName = ((arax_object_s) (((arax_proc_s *) (task->proc))->obj)).name;

    proc = (arax_proc_s *) task->proc;
    // Check if the peeked task can run in the current accelThread
    runnable = arax_proc_can_run_at(proc, accelThreadType);

    // Migrate VAQ, since the accelThread does not support this kernel
    if (runnable == 0) {
        #ifdef DEBUG_NN
        std::cerr << "4. " << __func__ << "Task State (" << taskName
                  << "):  Non Runnable " << T
                  << " , accel: " << th->getAccelConfig().name << std::endl;
        #endif
        return T;
    } else { // The current thread is able to execute the task
        #ifdef DEBUG_NN
        std::cerr << "4. " << __func__ << "Task State (" << taskName
                  << "): Runnable task " << RT
                  << " , accel: " << th->getAccelConfig().name << std::endl;
        #endif
        return RT;
    }
} // HeterogeneousScheduler::checkForRunnableTask

/*Used from VACBalancer to assign a VAQ to an accelthread*/
void HeterogeneousScheduler::assignVac(arax_vaccel_s *vac)
{
    int runnable = -1;
    arax_task_msg_s *task = 0;
    arax_proc_s *proc     = 0;
    vector<AccelConfig *> accelsOfAType;

    accelsOfAType = config->getAccelerators(vac->type);
    #ifdef DEBUG_NN
    std::cerr << " 1. VAC Balancer called assignVac!!!!!!!!!!!!!!!! "
              << std::endl;
    #endif
    for (int i = 0; i < accelsOfAType.size(); i++) {
        static auto paccel_rr = accelsOfAType.begin();

        // This will RR on the available physical accelerators
        if (paccel_rr == accelsOfAType.end()) {
            paccel_rr = accelsOfAType.begin();
        }
        #ifdef DEBUG_NN
        std::cerr << " 2. Current accelThread is " << (*paccel_rr)->name
                  << std::endl;
        #endif
        int task_state = checkForRunnableTask((*paccel_rr)->accelthread, vac);
        #ifdef DEBUG_NN
        std::cerr << "5. [" << __func__ << "] Task state: " << task_state
                  << " Accelerator: " << (*paccel_rr)->name << std::endl;
        #endif
        switch (task_state) {
            case RT: // Found a runnable task -> Assign it to an accelerator
                #ifdef DEBUG_NN
                std::cerr << "6. " << __func__
                          << " Found a runnable task -> Assign it to "
                          << (*paccel_rr)->name << std::endl;
                #endif
                arax_accel_add_vaccel((*paccel_rr)->arax_accel, vac);
                return;

                break;
            case T: // Found a Non runnable task -> Try the next accelerator
                #ifdef DEBUG_NN
                std::cerr << "6. " << __func__
                          << " Found a NON runnable task try next thread" << std::endl;
                #endif
                paccel_rr++;
                break;
            default: // No task
                #ifdef DEBUG_NN
                std::cerr << __func__ << " Empty VAQ -> Assign it randomly "
                          << (*paccel_rr)->name << std::endl;
                #endif
                // If there is an empty VAQ assign randomly to an accelThread
                arax_accel_add_vaccel((*paccel_rr)->arax_accel, vac);
                return;
        }
    }
    std::cerr << "NO Assignment!!! " << std::endl;
    abort();
} // HeterogeneousScheduler::assignVac

/*Used from accelThreads to assign a VAQ to an accelthread*/
arax_task_msg_s * HeterogeneousScheduler::selectTask(accelThread *th)
{
    auto &VACs = th->getAssignedVACs();

    #ifdef DEBUG_NN
    std::cerr << th->getAccelConfig().name << " calls [" << __func__ << "]: "
              << "VACs.size(): " << VACs.size() << std::endl;
    #endif
    if (VACs.size() == 0) {
        return 0;
    }

    auto rr_offset = (rr_counter[th] + 1) % VACs.size();
    auto rr_point  = VACs.begin() + rr_offset;
    auto itr       = rr_point;
    char *taskName = NULL;
    arax_task_msg_s *task;
    for (itr = rr_point; itr != VACs.end(); itr++) {
        int task_state = checkForRunnableTask(th, *itr);
        #ifdef DEBUG
        std::cerr << __func__ << " calls checkForRunnableTask" << std::endl;
        std::cerr << "[" << __func__ << "] Task state: " << task_state << std::endl;
        #endif
        switch (task_state) {
            case RT: // Found a runnable task -> Execute!
                #ifdef DEBUG_NN
                std::cerr << "3. [" << __func__ << "] found a runnable task execute!"
                          << std::endl;
                #endif
                task = (arax_task_msg_s *) utils_queue_pop(arax_vaccel_queue(*itr));
                arax_assert(task);
                th->migrateFromRemote(task);
                th->migrateToRemote(task);
                th->incMigrations();
                goto FOUND_TASK;
                break;
            case T: // Found a Non runnable task
                    // Delete VAQ (so VACBalancer will run again)
                #ifdef DEBUG_NN
                std::cerr << "3. [" << __func__ << "] found a NON runnable delete VAQ!"
                          << std::endl;
                #endif
                arax_accel_del_vaccel((*itr)->phys, *itr);
                arax_pipe_add_orphan_vaccel(th->getPipe(), *itr);
                break;
            default: // No task
                int used_slots = utils_queue_used_slots(arax_vaccel_queue(*itr));
                #ifdef DEBUG_NN
                std::cerr << "3. [" << __func__
                          << "] found No task (Vaq used slots: " << used_slots << ")!"
                          << std::endl;
                #endif
                #ifdef DEBUG
                std::cerr << "VAQ (" << *itr << ") has " << used_slots << " pending tasks"
                          << std::endl;
                #endif
                // There is no task continue RR to the other assigned VAQs
                break;
        }
    }

    for (itr = VACs.begin(); itr != rr_point; itr++) {
        switch (checkForRunnableTask(th, *itr)) {
            case RT: // Found a runnable task -> Execute!
                #ifdef DEBUG_NN
                std::cerr << "3. [" << __func__ << "] 2. found a runnable task execute!"
                          << std::endl;
                #endif
                task = (arax_task_msg_s *) utils_queue_pop(arax_vaccel_queue(*itr));
                th->migrateFromRemote(task);
                th->migrateToRemote(task);
                th->incMigrations();
                arax_assert(task);
                goto FOUND_TASK;
                break;
            case T: // Found a Non runnable task
                    // Delete VAQ (so VACBalancer will run again)
                #ifdef DEBUG_NN
                std::cerr << "3. [" << __func__ << "] 2. found a NON runnable delete VAQ!"
                          << std::endl;
                #endif
                arax_accel_del_vaccel((*itr)->phys, *itr);
                arax_pipe_add_orphan_vaccel(th->getPipe(), *itr);
                break;
            default: // No task
                int used_slots = utils_queue_used_slots(arax_vaccel_queue(*itr));
                #ifdef DEBUG_NN
                std::cerr << "3. [" << __func__
                          << "] 2. found No task (Vaq used slots: " << used_slots << ")!"
                          << std::endl;
                #endif
                #ifdef DEBUG
                int used_slots = utils_queue_used_slots(arax_vaccel_queue(*itr));
                std::cerr << "VAQ (" << *itr << ") has " << used_slots << " pending tasks"
                          << std::endl;
                #endif
                // There is no task continue RR to the other assigned VAQs
                break;
        }
    }
    return 0;

FOUND_TASK:
    rr_counter[th] = itr - VACs.begin();
    return task;
} // HeterogeneousScheduler::selectTask

REGISTER_SCHEDULER(HeterogeneousScheduler)
