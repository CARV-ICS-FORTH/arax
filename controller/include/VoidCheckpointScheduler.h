#ifndef VOID_CHECKPOINT_SCHEDULER
#define VOID_CHECKPOINT_SCHEDULER

#include "CheckpointScheduler.h"
using namespace ::std;

class VoidCheckpointScheduler : public CheckpointScheduler {
public:
    VoidCheckpointScheduler(string args);
    virtual ~VoidCheckpointScheduler();
    virtual void checkpointFrequency(accelThread *th); // determines when to checkpoint
};
#endif // ifndef VOID_CHECKPOINT_SCHEDULER
