#ifndef ACCELTHREAD
#define ACCELTHREAD
#include "arax_pipe.h"
#include "timers.h"
#include <atomic>
#include <map>
#include <ostream>
#include <pthread.h>
#include <set>
#include <typeinfo>
#include <vector>
class accelThread;
#include "Config.h"
#include "Factory.h"
#include "Scheduler.h"
#include "deviceBaseOps.h"
#include <unordered_map>
using namespace std;

#define VECTOR_WITH_DURATIONS_SIZE 30
class accelThread : public deviceBaseOps {
public:
    accelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
    void terminate();
    void joinThreads();
    arax_pipe_s* getPipe();
    size_t getServedTasks();
    virtual ~accelThread();
    virtual void printOccupancy();
    /* signal to exit*/
    volatile int stopExec;
    /*Getters*/
    arax_accel_type_e getAccelType();
    const vector<arax_vaccel_s *> &getAssignedVACs();
    /*returns Accelarator configuration specified from the configuration file*/
    AccelConfig &getAccelConfig();
    AccelConfig::JobPreference getJobPreference();
    AccelConfig::JobPreference getInitialJobPreference();
    void setJobPreference(AccelConfig::JobPreference jpref);
    void enable();
    void disable();
    arax_task_msg_s* getRunningTask();
    virtual size_t getAvailableSize(); // get accelerator's memory usage
    virtual size_t getTotalSize();     // get accelerator's total memory
    utils_queue_s* getFreeTaskQueue();
    bool isReadyToServe();
    accelThread* getVAQPhys2accelThread(arax_accel *);
    void addVAQPhys2accelThread(accelThread *, arax_accel *);
    // migrate data to rmt from shm (called from new thread before executing a
    // task)
    void migrateToRemote(arax_task_msg_s *);
    // migrate data from rmt to shm (called from old Thread that is not able to
    // run a task)
    void migrateFromRemote(arax_task_msg_s *);
    int getNumMigrations();
    void incMigrations();

private:
    std::atomic<bool> readyToServe;
    size_t served_tasks;
    pthread_mutex_t enable_mtx;
    /* thread*/
    pthread_t pthread;
    /* Accelerator configuration*/
    AccelConfig &accelConf;
    size_t initialAvailableSize; // < Memory used before serving tasks
    /*function to check if there are new VA queues*/
    void updateVirtAccels();
    size_t revision;
    /*function that performs the execution per thread*/
    friend void* workFunc(void *thread);
    /*vector with timers*/
    vector<Timers_s> statsVector;
    std::vector<arax_vaccel_s *> assignedVACs;
    /*Task that is currenlty executed to the accelerator*/
    arax_task_msg_s *runningTask;
    set<std::string> operations2Ignore;
    static unordered_map<arax_accel *, accelThread *> VAQPhys2accelThread;
    int migrations;

    /*#ifdef FREE_THREAD
     * utils_queue_s *freeTaskQueue; // a queue with tasks that should be ref_dec
     #endif
     */
protected:

    /**
     * Actually start thread.
     *
     * This should be callled at the end of the constructor of the subclass,
     * to ensure the subclass is fully initialized prior to thread execution.
     */
    void start();
    /* Arax Pipe */
    arax_pipe_s *v_pipe;
};

void printAccelThreadState(string currState, accelThread &currThread,
  arax_task_msg_s *arax_task, size_t outstanding);
extern Factory<accelThread, arax_pipe_s *, AccelConfig &> threadFactory;

/**
 * Helper function to register Accelerator Threads
 * Must be put in a cpp file.
 */
#define REGISTER_ACCEL_THREAD(CLASS)                                           \
    static Registrator<accelThread, CLASS, arax_pipe_s *, AccelConfig &> reg(    \
        threadFactory);
#endif // ifndef ACCELTHREAD
