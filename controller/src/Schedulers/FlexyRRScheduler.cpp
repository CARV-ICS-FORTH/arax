#include "FlexyRRScheduler.h"
#include "Utilities.h"
#include "utils/timer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
// #define All4Batch
#define BATCH_JOB  0
#define USERF_TASK 1
// #define FLEXY_DEBUG
#ifdef FLEXY_DEBUG
#define DEBUG_MSG(EXPR) std::cerr << EXPR << std::endl
#else
#define DEBUG_MSG(EXPR)
#endif
mutex myM;
using namespace ::std;
static const double sla = 200000; // us
std::map<int, JobMetrics *> job_metrics;
FlexyRRScheduler::FlexyRRScheduler(picojson::object args)
    : Scheduler(args), userRR(0), batchRR(0)
{
    jsonGetSafeOptional(args, "elastic", elastic, "expeted a boolean", true);
    jsonGetSafeOptional(args, "proactive", proactive, "expeted a boolean", true);
    //	proactAll4Batch = (kv.count("proactAll4Batch") && kv["proactAll4Batch"]
    // == "true");
    std::cerr << __func__ << "(elastic:" << elastic << ")\n";
    std::cerr << __func__ << "(proactive:" << proactive << ")\n";
    //	std::cerr << __func__ << "(proactAll4Batch:" << proactAll4Batch <<
    // ")\n";

    std::string sla_str;

    jsonGetSafe(args, "sla", sla_str, "file path");

    std::ifstream ifs(sla_str);
    std::string line;
    int line_n = 0;

    if (!ifs) {
        throw std::runtime_error("No SLA file \"" + sla_str
                + "\" could not be opened.");
    }
    while (std::getline(ifs, line)) {
        auto args = decodeArgs(line);
        if (!(args.count("id") && args.count("sla") &&
          args.count("over_threshold") && args.count("under_threshold")))
        {
            throw std::runtime_error("Sla description missing arguement at line "
                    + std::to_string(line_n));
        }
        line_n++;
        job_metrics[atoi(args["id"].c_str())] = new JobMetrics(
            atol(args["sla"].c_str()), atol(args["over_threshold"].c_str()),
            atol(args["under_threshold"].c_str()));
    }
}

FlexyRRScheduler::~FlexyRRScheduler(){ }

/*
 * For now i will use the customer id as a flag to specify if a task is
 * batch or user.
 */
bool isUserTask(arax_accel *accel)
{
    bool isUser =
      arax_vaccel_get_job_priority((arax_vaccel_s *) accel) == USERF_TASK;

    return isUser;
}

/**
 * Return first task found of \c type from \c vac_array of size \c vac_len
 * starting from \c index.
 */
std::pair<arax_vaccel_s *, arax_task_msg_s *> pickFirstTypeJobTaskRR(arax_accel_type_e type, arax_vaccel_s **vac_array,
  size_t vac_len, int &index, accelThread *th)
{
    arax_vaccel_s *vac;
    utils_queue_s *vaq;
    arax_task_msg_s *task = 0;

    for (size_t rr = 0; rr < vac_len; rr++) { // Find a user task
        DEBUG_MSG("Index: " << (index) % vac_len);
        vac = vac_array[(index) % vac_len];
        vaq = arax_vaccel_queue(vac);
        index++;
        if (vac->type == type &&
          arax_vaccel_test_set_assignee((arax_accel_s *) vac, th))
        {
            task = (arax_task_msg_s *) utils_queue_pop(vaq);
            if (task) {
                return std::pair<arax_vaccel_s *, arax_task_msg_s *>(vac, task);

                ;
            }
        }
    }
    return std::pair<arax_vaccel_s *, arax_task_msg_s *>(0, 0);
}

bool isOverSLA(arax_vaccel_s *accel)
{
    JobMetrics *jm = (JobMetrics *) arax_vaccel_get_meta(accel);

    if (!jm)
        return false;

    return jm->overSLA();
}

bool isUnderSLA(arax_vaccel_s *accel)
{
    JobMetrics *jm = (JobMetrics *) arax_vaccel_get_meta(accel);

    if (!jm)
        return false;

    return jm->underSLA();
}

void flexPolicyReactive(GroupConfig *group, arax_vaccel_s **arrayAllVAQs,
  size_t user_jobs, size_t batch_jobs)
{
    int over_sla =
      std::count_if(arrayAllVAQs, arrayAllVAQs + user_jobs, isOverSLA);
    int under_sla =
      std::count_if(arrayAllVAQs, arrayAllVAQs + user_jobs, isUnderSLA);

    DEBUG_MSG("overSLA:" << over_sla);
    DEBUG_MSG("underSLA:" << under_sla);
    // User facing
    if (over_sla) {
        DEBUG_MSG("I need more acceleration!");
        for (auto accel : group->getAccelerators()) {
            if (accel.second->accelthread->getInitialJobPreference() !=
              AccelConfig::BatchJob &&
              accel.second->accelthread->getJobPreference() !=
              AccelConfig::UserJob)
            {
                accel.second->accelthread->setJobPreference(AccelConfig::UserJob);
                DEBUG_MSG("Activating accelerator");
                break;
            }
        }
    }
    if (under_sla && !over_sla) {
        DEBUG_MSG("I need less acceleration!");
        for (auto accel : group->getAccelerators()) {
            if (accel.second->accelthread->getInitialJobPreference() !=
              AccelConfig::UserJob &&
              accel.second->accelthread->getJobPreference() ==
              AccelConfig::UserJob)
            {
                DEBUG_MSG("I need less acceleration!");
                accel.second->accelthread->setJobPreference(AccelConfig::BatchJob);
                break;
            }
        }
    }
    DEBUG_MSG("User tasks: " << user_jobs << "Batch tasks: " << batch_jobs);
    if (!user_jobs && batch_jobs) { // No user tasks so give all accelerators
                                    // except one to batch
        DEBUG_MSG("Batch off!");
        for (auto accel : group->getAccelerators()) {
            if (accel.second->accelthread->getInitialJobPreference() !=
              AccelConfig::UserJob)
            {
                accel.second->accelthread->setJobPreference(AccelConfig::BatchJob);
            }
        }
    }
} // flexPolicyReactive

void flexPolicyProactive(GroupConfig *group, arax_vaccel_s **arrayAllVAQs,
  size_t user_jobs, size_t batch_jobs,
  size_t *usedSlots)
{
    arax_vaccel_s *vac;
    utils_queue_s *vaq;
    double duration = 0;
    double uSlots   = 0;
    // latency of the last task added
    double qmax = 0;

    for (size_t v = 0; v < user_jobs; v++) {
        // get a VAQ from the existing
        vac = arrayAllVAQs[v];
        vaq = arax_vaccel_queue(vac);
        JobMetrics *jm = (JobMetrics *) arax_vaccel_get_meta(vac);
        if (jm) {
            //  Last tasks predicted latency = Average latency * length of VAQ
            qmax     += jm->getAverageDuration() * utils_queue_used_slots(vaq);
            duration += jm->getAverageDuration();
            uSlots   += utils_queue_used_slots(vaq);
        } else {
            qmax     += sla * utils_queue_used_slots(vaq);
            duration += sla;
            uSlots   += utils_queue_used_slots(vaq);
        }
    }
    *usedSlots = uSlots;
    int all_accels = group->countAccelerators();
    // needed accelerators to serve the existing load
    int req_accels = std::ceil(qmax / sla);

    #ifdef All4Batch
    // if (proactAll4Batch){
    if (!req_accels && user_jobs) { // all gpus to batch
        req_accels = 1;
    }
    // }
    // else{
    #else
    if (!req_accels) { // all gpus except one to batch
        req_accels = 1;
    }
    // }
    #endif // ifdef All4Batch
    // if the number of the requested accels is greater than the existing
    if (req_accels > all_accels)
        req_accels = all_accels;
    for (auto accel : group->getAccelerators()) {
        // if there is SLA violations
        if (req_accels) {
            // give an accelerator to user jobs
            accel.second->accelthread->setJobPreference(AccelConfig::UserJob);
            // reduce the number of the avaliable accels
            req_accels--;
        } else {
            // provide accel to batch
            accel.second->accelthread->setJobPreference(AccelConfig::BatchJob);
        }
    }
} // flexPolicyProactive

/**
 * Performs the following in that order:
 * 1) Split current vaqs based on type (user/batch)
 * 2) Return a runnable task from a User job using round robin.
 * 3) Return a runnable task from a Batch job using round robin.
 * 4) If no suitable task is found return null.
 **/
arax_task_msg_s * FlexyRRScheduler::selectTask(accelThread *th)
{
    arax_vaccel_s **arrayAllVAQs = (arax_vaccel_s **) th->getAllVirtualAccels();
    arax_task_msg_s *task;
    std::pair<arax_vaccel_s *, arax_task_msg_s *> selected;
    arax_accel_s *phys = th->getAccelConfig().arax_accel;
    int numOfVAQs      = th->getNumberOfVirtualAccels();
    size_t usedSlots   = 0;
    // put user facing jobs in the beginning on array with all VAQs
    auto split =
      std::stable_partition(arrayAllVAQs, arrayAllVAQs + numOfVAQs,
        isUserTask); // Separate user and batch tasks
    // number of user_jobs
    size_t user_jobs  = std::distance(arrayAllVAQs, split);
    size_t batch_jobs = numOfVAQs - user_jobs;

    DEBUG_MSG("User Jobs:" << user_jobs << " Batch Jobs:" << batch_jobs
                           << " All Jobs:" << numOfVAQs);
    if (elastic) {
        if (proactive) {
            flexPolicyProactive(group, arrayAllVAQs, user_jobs, batch_jobs,
              &usedSlots);
        } else {
            flexPolicyReactive(group, arrayAllVAQs, user_jobs, batch_jobs);
        }
    }
    if (th->getJobPreference() & AccelConfig::UserJob) { // I will serve User Jobs
        selected =
          pickFirstTypeJobTaskRR(phys->type, arrayAllVAQs, user_jobs, userRR, th);
        task = selected.second;
        if (task) {
            utils_timer_set(task->stats.task_duration_without_issue, start);
            /*print the number of accelerator that are used*/
            //			printAccelThreadState(to_string(selected.first->priority),
            // *th, task, usedSlots+1);
            task->stats.usedSlots = usedSlots;
            DEBUG_MSG("Picked:User " << (void *) task);
            return task;
        }
        DEBUG_MSG("No User tasks");
    } else {
        AccelConfig &accelConf = th->getAccelConfig();
        DEBUG_MSG("Skipping User jobs");
        arax_pipe_add_task(th->getPipe(), accelConf.type, th);
    }
    if (th->getJobPreference()
      & AccelConfig::BatchJob) // I will serve Batch Jobs
    {
        selected = pickFirstTypeJobTaskRR(phys->type, arrayAllVAQs + user_jobs,
            batch_jobs, batchRR, th);
        task = selected.second;
        if (task) {
            utils_timer_set(task->stats.task_duration_without_issue, start);
            /*print the number of accelerator that are used*/
            //			printAccelThreadState(to_string(selected.first->priority),
            // *th, task, usedSlots+1);
            task->stats.usedSlots = usedSlots;
            DEBUG_MSG("Picked:Batch");
            return task;
        }
        DEBUG_MSG("No Batch tasks");
    } else {
        AccelConfig &accelConf = th->getAccelConfig();
        DEBUG_MSG("Skipping Batch jobs");
        arax_pipe_add_task(th->getPipe(), accelConf.type, th);
    }
    // Didn't run a thing
    DEBUG_MSG("Picked:None");
    return 0;
} // FlexyRRScheduler::selectTask

void FlexyRRScheduler ::postTaskExecution(accelThread *th,
  arax_task_msg_s *                                    task)
{
    int cid;
    JobMetrics *jm =
      (JobMetrics *) arax_vaccel_get_meta((arax_vaccel_s *) task->accel);

    if (!jm) {
        cid = arax_vaccel_get_cid((arax_vaccel_s *) task->accel);
        jm  = job_metrics[cid];
        if (!jm) {
            // cout<<"sla.spec does not contain enough lines!! EXIT"<<endl;
            exit(1);
            // cout<<"JM: "<<jm<<endl;
        }
        arax_vaccel_set_meta((arax_vaccel_s *) task->accel, jm);
    }
    jm->addDuration(
        utils_timer_get_duration_us(task->stats.task_duration_without_issue));
    /*stop meassuring task exec (without queueing). Used from Flexy policy*/
    utils_timer_set(task->stats.task_duration_without_issue, stop);
    /*stop meassuring task exec (with queueing). Used from Flexy policy*/
    utils_timer_set(task->stats.task_duration, stop);
    //	printAccelThreadState("-1", *th, task, task->stats.usedSlots);
    // cout<<"Task time
    // :"<<(utils_timer_get_duration_us(task->stats.task_duration)/1000)<<"
    // average:     " << jm->getAverageDuration() <<endl;   DEBUG_MSG("Task
    // time:"
    // << (utils_timer_get_duration_us(task->stats.task_duration)/1000)<<" average:
    // " << jm->getAverageDuration() );
}

std::ostream &operator << (std::ostream &os, const JobMetrics &jm)
{
    os << "Sla:" << jm.sla << " OverThreshold:" << jm.over_sla_threshold
       << " UnderThreshold:" << jm.under_sla_threshold
       << " Average:" << jm.getAverageDuration() << std::endl;
    return os;
}

REGISTER_SCHEDULER(FlexyRRScheduler)
