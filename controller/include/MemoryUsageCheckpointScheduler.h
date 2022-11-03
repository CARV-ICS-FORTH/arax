#ifndef MEMORY_CHECKPOINT_SCHEDULER
#define MEMORY_CHECKPOINT_SCHEDULER

#include "CheckpointScheduler.h"

class MemoryUsageCheckpointScheduler : public CheckpointScheduler {
public:
    MemoryUsageCheckpointScheduler(std::string args);
    virtual ~MemoryUsageCheckpointScheduler();
    virtual size_t getCurrentMemoryUsage();
    virtual void resetCurrentMemoryUsage();
    virtual void checkpointFrequency(); // determines when to checkpoint
private:
    size_t currentMemoryUsage;
};
#endif // ifndef MEMORY_CHECKPOINT_SCHEDULER
