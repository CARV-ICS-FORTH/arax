#ifndef ELASTIC_SCHEDULER
#define ELASTIC_SCHEDULER
#include "Scheduler.h"
#include <map>
#include <vector>

class ElasticScheduler : public Scheduler {
    // Return types of checkForRunnableTask (RunnableTask, Task, NoTask)
    typedef enum State { RT, T, NT } State;

public:
    ElasticScheduler(picojson::object args);
    virtual ~ElasticScheduler();

    virtual void assignVac(arax_vaccel_s *vac);

    /*Select a task from all the VAQs that exist in the system  */
    virtual arax_task_msg_s* selectTask(accelThread *threadPerAccel);

    ElasticScheduler::State checkForRunnableTask(accelThread *th,
      arax_vaccel_s *                                         vaq);
    arax_vaccel_s *vaq4migration;

private:
    std::map<accelThread *, int> rr_counter;
};

#endif // ifndef ELASTIC_SCHEDULER
