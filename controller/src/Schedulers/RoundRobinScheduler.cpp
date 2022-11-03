#include <iostream>
using namespace ::std;
#include "RoundRobinScheduler.h"
RoundRobinScheduler::RoundRobinScheduler(picojson::object args)
    : Scheduler(args) {}

RoundRobinScheduler::~RoundRobinScheduler() {}

void RoundRobinScheduler::assignVac(arax_vaccel_s *vac) {
  vector<AccelConfig *> accelsOfAType;
  accelsOfAType = config->getAccelerators(vac->type);

  // MP: not sure if we need it
  // if (accelsOfAType.size() == 0)
  //  return 0;
  // This will RR on the available physical accelerators
  static auto paccel_rr = accelsOfAType.begin();
  if (paccel_rr == accelsOfAType.end()) {
    paccel_rr = accelsOfAType.begin();
  }
  arax_accel_add_vaccel((*paccel_rr)->arax_accel, vac);
  paccel_rr++;
}

arax_task_msg_s *RoundRobinScheduler::selectTask(accelThread *th) {
  auto &VACs = th->getAssignedVACs();

  if (VACs.size() == 0)
    return 0;

  auto rr_offset = (rr_counter[th] + 1) % VACs.size();
  auto rr_point = VACs.begin() + rr_offset;
  auto itr = rr_point;
  arax_task_msg_s *task = 0;

  for (itr = rr_point; itr != VACs.end(); itr++) {
    task = (arax_task_msg_s *)utils_queue_pop(arax_vaccel_queue(*itr));
    if (task)
      goto FOUND_TASK;
  }

  for (itr = VACs.begin(); itr != rr_point; itr++) {
    task = (arax_task_msg_s *)utils_queue_pop(arax_vaccel_queue(*itr));
    if (task)
      goto FOUND_TASK;
  }

  return 0;

FOUND_TASK:
  rr_counter[th] = itr - VACs.begin();
  return task;
}

REGISTER_SCHEDULER(RoundRobinScheduler)
