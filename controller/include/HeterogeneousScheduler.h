#ifndef HETEROGENEOUS_SCHEDULER
#define HETEROGENEOUS_SCHEDULER
#include "Scheduler.h"
#include <map>
#include <vector>

class HeterogeneousScheduler : public Scheduler {
    // Return types of checkForRunnableTask (RunnableTask, Task, NoTask)
    typedef enum State { RT, T, NT } State;

public:
    HeterogeneousScheduler(picojson::object args);
    virtual ~HeterogeneousScheduler();

    virtual void assignVac(arax_vaccel_s *vac);

    /*Select a task from all the VAQs that exist in the system  */
    virtual arax_task_msg_s* selectTask(accelThread *threadPerAccel);

    HeterogeneousScheduler::State checkForRunnableTask(accelThread *th,
      arax_vaccel_s *                                               vaq);

private:
    std::map<accelThread *, int> rr_counter;
};

#endif // ifndef HETEROGENEOUS_SCHEDULER
