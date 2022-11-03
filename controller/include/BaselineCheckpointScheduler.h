#ifndef BASELINE_CHECKPOINT_SCHEDULER
#define BASELINE_CHECKPOINT_SCHEDULER

#include "CheckpointScheduler.h"
using namespace ::std;

class BaselineCheckpointScheduler : public CheckpointScheduler {

public:
  BaselineCheckpointScheduler(string args);
  virtual ~BaselineCheckpointScheduler();
  virtual void
  setCheckFreq(arax_task_msg_s *task); // increases the batch counter
  virtual void resetCheckFreq();       // reset the batch counter (set to 0)
  virtual void
  checkpointFrequency(accelThread *th); // determines when to checkpoint
private:
  size_t batchTaskCountCH; // counter for batch tasks used for checkpoint
};
#endif
