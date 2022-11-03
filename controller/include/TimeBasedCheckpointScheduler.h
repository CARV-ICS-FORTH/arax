#ifndef TIME_CHECKPOINT_SCHEDULER
#define TIME_CHECKPOINT_SCHEDULER
#include "CheckpointScheduler.h"
class TimeBasedCheckpointScheduler : public CheckpointScheduler {
public:
  TimeBasedCheckpointScheduler(std::string args);
  virtual ~TimeBasedCheckpointScheduler();
  virtual void setCheckFreq(arax_task_msg_s *task); // meassure task exec time
  virtual void resetCheckFreq(); // reset the duration counter (set to 0)
  virtual void
  checkpointFrequency(accelThread *th); // determines when to checkpoint
private:
  double allTaskDur; // total task duration in ms
};
#endif
