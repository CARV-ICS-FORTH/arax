#ifndef ARAXROUND_ROBIN_SCHEDULER
#define ARAXROUND_ROBIN_SCHEDULER
#include "Scheduler.h"
#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

class LoadBalanceScheduler : public Scheduler {
public:
  LoadBalanceScheduler(picojson::object args);
  virtual ~LoadBalanceScheduler();
  /*Pick a Virtual Accelerator Queue(VAQ) from all avaliable VAQs that exist in
   * the system*/
  virtual utils_queue_s *selectVirtualAcceleratorQueue(accelThread *th);
  /*Select a task from all the VAQs that exist in the system  */
  virtual arax_task_msg_s *selectTask(accelThread *th);
  typedef std::pair<accelThread *, int> MyPairType;
  struct CompareSecond {
    bool operator()(const MyPairType &left, const MyPairType &right) const {
      return left.second < right.second;
    }
  };
  accelThread *getMin(std::unordered_map<accelThread *, int> mymap);
  void assignVaqToTh(accelThread *th);
  virtual void accelThreadSetup(accelThread *th);

private:
  unordered_map<accelThread *, int> physAccelJobCounter;
  map<arax_accel_s *, int> virtualQueueIndex;
  mutex mutexForThAndJobs;
};

#endif
