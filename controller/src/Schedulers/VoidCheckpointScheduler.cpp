#include "VoidCheckpointScheduler.h"
#include <iostream>
#define NUMOFBATCHTASKS 10

using namespace ::std;
VoidCheckpointScheduler::VoidCheckpointScheduler(string args)
    : CheckpointScheduler(args){ }

VoidCheckpointScheduler::~VoidCheckpointScheduler(){ }

void VoidCheckpointScheduler ::checkpointFrequency(accelThread *th){ }

REGISTER_CHECKPOINT(VoidCheckpointScheduler)
