#ifndef ARAXROUND_ROBIN_SCHEDULER
#define ARAXROUND_ROBIN_SCHEDULER
#include "Scheduler.h"
#include <atomic>
#include <map>
#include <vector>

/*
 * FlexyRRScheduler:
 * One scheduler to rule them all
 * Its flexible and RoundRobin.
 */
class FlexyRRScheduler : public Scheduler {
public:
  /*
   * Constructor.
   *
   * args contains of comma separated key:value pairs in the following
   * format:
   *
   * key1:val1,key2:val2
   *
   * Acceptable Keys:
   * elastic:true|false	// Enable/disable elastic mode
   * sla:<file>			// Use <file> to load SLA and other settings for
   * jobs
   */
  FlexyRRScheduler(picojson::object);
  virtual ~FlexyRRScheduler();
  bool proactAll4Batch; // True if args contains proactAll4Batch:true
  /*Select a task from all the VAQs that exist in the system  */
  virtual arax_task_msg_s *selectTask(accelThread *threadPerAccel);
  virtual void postTaskExecution(accelThread *th, arax_task_msg_s *task);

private:
  int userRR;     // Round Robin Counter for User Facing Tasks
  int batchRR;    // Round Robin Counter for Batch Tasks
  bool elastic;   // True if args contains elastic:true
  bool proactive; // True if args contains proactive:true
  bool resetPol;  // True if args contains reset:true
};

#define JOB_METRIC_SAMPLES 2

class JobMetrics {
private:
  uint64_t counter;
  double durations[JOB_METRIC_SAMPLES];
  double cumulative_task_ttc; // Time To Completion
  double sla;                 // Sla in us
  int over_sla_threshold;
  int under_sla_threshold;
  int over_sla;
  int under_sla;
  //		bool isUser_Facing;

public:
  friend std::ostream &operator<<(std::ostream &os, const JobMetrics &jm);
  JobMetrics(double sla = 200000, double over_sla_threshold = 10,
             double under_sla_threshold = 10)
      : counter(0), cumulative_task_ttc(0), sla(sla),
        over_sla_threshold(over_sla_threshold),
        under_sla_threshold(under_sla_threshold), over_sla(0), under_sla(0) {
    std::fill(durations, durations + JOB_METRIC_SAMPLES, 0);
  }

  void addDuration(double dur) {
    uint64_t index = __sync_add_and_fetch(&counter, 1);
    index %= JOB_METRIC_SAMPLES;
    cumulative_task_ttc -= durations[index];
    durations[index] = dur;
    cumulative_task_ttc += dur;
    if (dur > sla) {
      __sync_add_and_fetch(&over_sla, 1);
      under_sla = 0;
    } else if (dur < sla) {
      __sync_add_and_fetch(&under_sla, 1);
      over_sla = 0;
    }
  }

  bool overSLA() {
    // if(!counter)
    //	return true;
    return over_sla >= over_sla_threshold;
  }

  bool underSLA() {
    // if(!counter)
    //       return false;
    return under_sla >= under_sla_threshold;
  }
  /*
     void setUser(bool user)
     {
     isUser_Facing = user;
     }
     */
  void reset() {
    counter = 0;
    cumulative_task_ttc = 0;
    over_sla = 0;
    under_sla = 0;
  }
  double getAverageDuration() const {
    if (counter < JOB_METRIC_SAMPLES) {
      if (!counter)
        return 0;
      else
        return cumulative_task_ttc / counter;
    } else
      return cumulative_task_ttc / JOB_METRIC_SAMPLES;
  }
  double getsla() { return sla; }
  double getcumulativettc() { return cumulative_task_ttc; }
};

std::ostream &operator<<(std::ostream &os, const JobMetrics &jm);

#endif
