#ifndef CHECKPOINT_SCHEDULER
#define CHECKPOINT_SCHEDULER

class CheckpointScheduler;

#include "Config.h"
#include "Factory.h"
#include "accelThread.h"
#include "core/arax_data.h"

class CheckpointScheduler {
public:
  CheckpointScheduler(std::string args);
  virtual void checkpoint1Task(arax_task_msg_s *task, accelThread *th);
  virtual void checkpointAllActiveTasks(accelThread *th);
  virtual void checkpointFrequency(accelThread *th);
  virtual void setCheckFreq(arax_task_msg_s *task);
  virtual void resetCheckFreq();
  virtual ~CheckpointScheduler();

protected:
  Config *config;
  std::vector<arax_data_s *> araxDataCH; // All arax data to checkpoint
};
extern Factory<CheckpointScheduler, std::string> checkpointSchedulerFactory;

#define REGISTER_CHECKPOINT(CLASS)                                             \
  static Registrator<CheckpointScheduler, CLASS, std::string> reg(             \
      checkpointSchedulerFactory);

#endif
