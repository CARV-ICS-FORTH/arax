#include "BaselineCheckpointScheduler.h"
#include <iostream>

// determines the nunmber of task until checkpoint
#define NUMOFBATCHTASKS 2

using namespace ::std;
BaselineCheckpointScheduler::BaselineCheckpointScheduler(string args)
    : CheckpointScheduler(args) {
  this->batchTaskCountCH = 0;
}

BaselineCheckpointScheduler::~BaselineCheckpointScheduler() {}

void BaselineCheckpointScheduler ::setCheckFreq(arax_task_msg_s *task) {
  batchTaskCountCH++;
}

void BaselineCheckpointScheduler ::resetCheckFreq() {
  this->batchTaskCountCH = 0;
}

void BaselineCheckpointScheduler ::checkpointFrequency(accelThread *th) {
  cerr << "Task until checkpoint: " << NUMOFBATCHTASKS - batchTaskCountCH
       << endl;
  // When the number of executed batch task reaches a certain value we
  // perform a checkpoint for all pending tasks
  if (batchTaskCountCH == NUMOFBATCHTASKS) {
    cerr << "Before checkpoint" << endl;
    checkpointAllActiveTasks(th);
    cerr << "After checkpoint" << endl;
    resetCheckFreq();
  }
}

REGISTER_CHECKPOINT(BaselineCheckpointScheduler)
