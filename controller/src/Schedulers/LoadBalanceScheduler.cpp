#include <iostream>
// using namespace::std;
#include "LoadBalanceScheduler.h"
#include <algorithm>
#include <assert.h>
#include <queue>
#include <string.h>
#include <vector>

LoadBalanceScheduler::LoadBalanceScheduler(picojson::object args)
    : Scheduler(args){ }

LoadBalanceScheduler::~LoadBalanceScheduler(){ }

accelThread * LoadBalanceScheduler::getMin(unordered_map<accelThread *, int> mymap)
{
    pair<accelThread *, int> min =
      *min_element(mymap.begin(), mymap.end(), CompareSecond());

    assert(min.first != NULL);
    return min.first;
}

void LoadBalanceScheduler::accelThreadSetup(accelThread *th)
{
    physAccelJobCounter.insert({ th, 0 });
}

utils_queue_s * LoadBalanceScheduler::selectVirtualAcceleratorQueue(accelThread *th)
{
    return 0;
}

/**
 * Find all the vaqs of the system and assign them to the th with the min jobs.
 *
 * Inputs: an accelerator th
 * Outputs: nothing
 */
void LoadBalanceScheduler::assignVaqToTh(accelThread *th)
{
    int numOfVAQs = th->getNumberOfVirtualAccels();
    arax_accel **arrayAllVAQs = th->getAllVirtualAccels();

    /*iterate through all the vaqs of the system*/
    for (int i = 0; i < numOfVAQs; i++) {
        mutexForThAndJobs.lock();
        /*tmpth is the th with the min jobs assigned to him*/
        accelThread *tmpth = getMin(physAccelJobCounter);
        void *thread       = arax_vaccel_get_assignee((arax_accel_s *) arrayAllVAQs[i]);
        /*if the vaq is already assigned to a th, then continue to the next*/
        if (thread) {
            /*set physical accelator, of tasks that belongs to vaq that is already
             * assigned to th, to arax_accel_s*/
            arax_accel_set_physical(
                (arax_accel_s *) arrayAllVAQs[i],
                ((accelThread *) thread)->getAccelConfig().arax_accel);
            mutexForThAndJobs.unlock();
            continue;
        } else if (arax_vaccel_test_set_assignee((arax_accel_s *) arrayAllVAQs[i],
          tmpth))
        {
            /*set physical accelator to arax_accel_s*/
            arax_accel_set_physical((arax_accel_s *) arrayAllVAQs[i],
              tmpth->getAccelConfig().arax_accel);
            /*increase the jobs of the th*/
            physAccelJobCounter[tmpth]++;
        }
        mutexForThAndJobs.unlock();
    }
}

/**
 * find all the vaqs assigned to the th, and then with round robin scheduling
 * policy pop task from that vaq
 *
 * Inputs: an accelerator th
 * Outputs: a arax_task_msg_s. a task from the vaq
 */
arax_task_msg_s * LoadBalanceScheduler::selectTask(accelThread *th)
{
    assignVaqToTh(th);

    /*Task poped from that VAQ*/
    arax_task_msg_s *arax_task;
    /*VAQ that the task is going to be poped*/
    utils_queue_s *selectedVAQ;
    int allVaqs;
    arax_accel **arrayAllVAQs = th->getAllVirtualAccels();

    /* find the vaqs */
    for (allVaqs = 0; allVaqs != th->getNumberOfVirtualAccels(); allVaqs++) {
        /* find the vaqs assigned to th */
        if (arax_vaccel_get_assignee((arax_accel_s *) arrayAllVAQs[allVaqs]) == th) {
            selectedVAQ = arax_vaccel_queue((arax_vaccel_s *) arrayAllVAQs[allVaqs]);

            /*If there is no VAQ*/
            if (!selectedVAQ)
                continue;
            /*take the task from that queue*/
            arax_task = (arax_task_msg_s *) utils_queue_pop(selectedVAQ);

            /*Pop a task*/
            if (arax_task)
                return arax_task;
        }
    }
    return 0;
}

REGISTER_SCHEDULER(LoadBalanceScheduler)
