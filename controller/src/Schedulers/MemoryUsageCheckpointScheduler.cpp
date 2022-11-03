#include "MemoryUsageCheckpointScheduler.h"
#include <iostream>
using namespace ::std;
MemoryUsageCheckpointScheduler::MemoryUsageCheckpointScheduler(string args)
    : CheckpointScheduler(args) {}
MemoryUsageCheckpointScheduler::~MemoryUsageCheckpointScheduler() {}

size_t MemoryUsageCheckpointScheduler ::getCurrentMemoryUsage() { return 0; }

void MemoryUsageCheckpointScheduler ::resetCurrentMemoryUsage() {}
void MemoryUsageCheckpointScheduler ::checkpointFrequency() {}
REGISTER_CHECKPOINT(MemoryUsageCheckpointScheduler)
