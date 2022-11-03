#ifndef ARAXROUND_ROBIN_SCHEDULER
#define ARAXROUND_ROBIN_SCHEDULER
#include "Core/Scheduler.h"
#include <map>
#include <vector>

class RoundRobinScheduler : public Scheduler {
public:
    RoundRobinScheduler(picojson::object args);
    virtual ~RoundRobinScheduler();

    virtual void assignVac(arax_vaccel_s *vac);

    /*Select a task from all the VAQs that exist in the system  */
    virtual arax_task_msg_s* selectTask(accelThread *threadPerAccel);

private:
    std::map<accelThread *, int> rr_counter;
};

#endif // ifndef ARAXROUND_ROBIN_SCHEDULER
