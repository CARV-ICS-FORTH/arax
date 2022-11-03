#ifndef ARAXSCHEDULER
#define ARAXSCHEDULER

class Scheduler;

#include "Config.h"
#include "Factory.h"
#include "accelThread.h"
class Scheduler {
public:
  Scheduler(picojson::object args);
  void setGroup(GroupConfig *group);
  void setConfig(Config *config);
  virtual ~Scheduler();
  virtual void assignVac(arax_vaccel_s *vac) = 0;
  virtual arax_task_msg_s *selectTask(accelThread *th) = 0;
  virtual void postTaskExecution(accelThread *th, arax_task_msg_s *task);
  /**
   * Perform accelThread specific setup for this scheduler.
   * \param th accelThread instance.
   */
  virtual void accelThreadSetup(accelThread *th);

protected:
  GroupConfig *group;
  Config *config;
};

extern Factory<Scheduler, picojson::object> schedulerFactory;

/**
 * Helper function to register Accelerator Threads
 * Must be put in a cpp file.
 */
#define REGISTER_SCHEDULER(CLASS)                                              \
  static Registrator<Scheduler, CLASS, picojson::object> reg(schedulerFactory);
#endif
