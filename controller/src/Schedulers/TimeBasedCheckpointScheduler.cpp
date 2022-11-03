#include "TimeBasedCheckpointScheduler.h"
#include "utils/timer.h"
#include <iostream>

using namespace ::std;

// determines the time until next checkpoint
// has to be in ms
#define TIMETOCHECKPOINT 10

TimeBasedCheckpointScheduler::TimeBasedCheckpointScheduler(string args)
    : CheckpointScheduler(args)
{
    this->allTaskDur = 0;
}

TimeBasedCheckpointScheduler::~TimeBasedCheckpointScheduler(){ }

void TimeBasedCheckpointScheduler ::setCheckFreq(arax_task_msg_s *task)
{
    // get last tasks duration in nano seconds
    double taskDur =
      utils_timer_get_duration_us(task->stats.task_duration_without_issue);
    // convert taskDur in milliseconds
    double taskDurInMS = taskDur / 1000000;

    allTaskDur += taskDurInMS;

    cerr << "Last task duration: " << fixed << taskDur / 1000000 << " ms" << endl;
    cerr << "Total tasks duration: " << fixed << allTaskDur << " ms" << endl;
}

void TimeBasedCheckpointScheduler ::resetCheckFreq(){ allTaskDur = 0; }

void TimeBasedCheckpointScheduler ::checkpointFrequency(accelThread *th)
{
    cerr << "Time untill checkpoint: " << TIMETOCHECKPOINT - allTaskDur << endl;
    if (allTaskDur >= TIMETOCHECKPOINT) {
        cerr << "Before checkpoint" << endl;
        checkpointAllActiveTasks(th);
        cerr << "After checkpoint" << endl;
        resetCheckFreq();
    }
}

REGISTER_CHECKPOINT(TimeBasedCheckpointScheduler)
