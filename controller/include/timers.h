#ifndef TIMERS_H
#define TIMERS_H
#include <chrono>
#include <ctime>

using namespace ::std;

/*Timers per job*/
struct Timers_s
{
    int                      jobId;
    int                      taskId;
    double                   elapsed_H2D; // time for copying data from Host2Device
    chrono::duration<double>
                             elapsed_D2H; // time for copying data from Device2Host
    double
                             elapsed_Malloc; // time for allocating memory to device (inputs+outputs)
    chrono::duration<double> elapsed_Free;   // time for free resources
    chrono::duration<double> elapsed_ExecT;  // time for excution time
    chrono::duration<double>
                             elapsed_mapping_VAQ_2_PA;  // time for scheduling decsion
    double                   avgTaskSchedulingDuration; // AVG scheduling decision per task
    chrono::time_point<chrono::system_clock>
                             startTimeTask; // point of time that a task has started
    chrono::time_point<chrono::system_clock>
                             endTimeTask; // point of time that a task has ended
};
#if 0
/*Printing results from timers */
static void printStats(vector<Timers> stats, map<string, Timers>::iterator timersIt)
{
    cout << "-----------Printing results--------------" << endl << endl;
    for (timersIt = stats.begin(); timersIt != stats.end(); ++timersIt) {
        cout << "JobID:                 " << timersIt->first << endl
             << "TaskID:                " << timersIt->second.taskId << endl
             << "Data transfer H2D:     " << timersIt->second.elapsed_H2D.count() << endl
             << "Data transfer D2H:     " << timersIt->second.elapsed_D2H.count() << endl
             << "Free resources:        " << timersIt->second.elapsed_D2H.count() << endl
             << "Execution time:        " << timersIt->second.elapsed_ExecT.count() << endl
             << "Mapping VAQ2 P.A.:     " << timersIt->second.elapsed_mapping_VAQ_2_PA.count() << endl
             << "AVG task scheduling:   " << timersIt->second.avgTaskSchedulingDuration << endl;
        // << "Start time:            "<< timersIt->second.startTimeTask.count() << endl
        // << "End time:              "<< timersIt->second.endTimeTask.count() << endl;
    }
    cout << endl << "-----------End printing results--------------" << endl;
}

#endif // if 0

#endif // ifndef TIMERS_H
