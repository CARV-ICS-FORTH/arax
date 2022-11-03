#include <iostream>
#include <mutex>

//#define DEBUG_ELASTICITY
using namespace ::std;
#include "ElasticScheduler.h"
ElasticScheduler::ElasticScheduler(picojson::object args) : Scheduler(args) {}

ElasticScheduler::~ElasticScheduler() {}

/*Used from VACBalancer to assign a VAQ to an accelthread*/
void ElasticScheduler::assignVac(arax_vaccel_s *vac) {
  //  std::cerr << "Call " << __func__ << std::endl;
  arax_task_msg_s *task = 0;
  arax_proc_s *proc = 0;
  vector<AccelConfig *> accelsOfAType;
  // All physical accelerators (accelThreads)
  accelsOfAType = config->getAccelerators(vac->type);
  static auto paccel_rr = accelsOfAType.begin();
  // std::cerr << "accelsOfAType.begin: " << &(*(accelsOfAType.begin()))
  //           << " end: " << &(*(accelsOfAType.end())) << std::endl;
  //   This will RR on the available physical accelerators
  if (paccel_rr == accelsOfAType.end()) {
    paccel_rr = accelsOfAType.begin();
  }
#ifdef DEBUG_ELASTICITY
  std::cerr << "1. Received VAC: " << vac << " , name: " << (*paccel_rr)->name
            << std::endl;
  std::cerr << "VAC type: " << vac->type << std::endl;
#endif
  // Get the thread
  accelThread *th = (*paccel_rr)->accelthread;
  // Get current assigned VAQs
  auto &VACs = th->getAssignedVACs();

  // If there are VAQs
  if (VACs.size() != 0) {
#ifdef DEBUG_ELASTICITY
    std::cerr << "2. (ELASTICITY) Accelerator " << th->getAccelConfig().name
              << " has ALREADY " << VACs.size() << " VAQs: " << std::endl;
#endif
    // TMP accelerator
    auto tmpAccel = paccel_rr;
    // Use the next accelerator
    tmpAccel++;
    // Start from the beggining, in case that are no other physical
    if (tmpAccel == accelsOfAType.end()) {
      tmpAccel = accelsOfAType.begin();
    }
    // Migrate and Delete ALREADY assigned VACs (remove load from the new accel)
    for (auto &mig : VACs) {
      arax_vaccel_s *freeVaq = (*tmpAccel)->arax_accel->free_vaq;
      arax_vaccel_s *freeVaq2 = (*paccel_rr)->arax_accel->free_vaq;
      // If the VAC are different than Free VAQs, perform migration
      if (mig != freeVaq && mig != freeVaq2) {
#ifdef DEBUG_ELASTICITY
        std::cerr << "Move VAQ " << mig << " from " << th->getAccelConfig().name
                  << " to " << ((*tmpAccel)->accelthread)->getAccelConfig().name
                  << std::endl;
#endif
        vaq4migration = mig;
        // Delete migrated VAQs from the current accelerator
        arax_accel_del_vaccel((*paccel_rr)->arax_accel, mig);

        // Move previous assigned VAQs to the other accelerator
        arax_accel_add_vaccel((*tmpAccel)->arax_accel, mig);
        // if VA for migration found: EXIT
        break;
      }
    }
#ifdef DEBUG_ELASTICITY
    std::cerr << "Move New VAC " << vac << " to"
              << (*paccel_rr)->accelthread->getAccelConfig().name << std::endl;
#endif
    // Assign the new VAQ to the new accelerator
    arax_accel_add_vaccel((*paccel_rr)->arax_accel, vac);
    // continue RR
    paccel_rr++;
  } else {
#ifdef DEBUG_ELASTICITY
    std::cerr << "3. (NO ELASTICITY) Accelerator " << th->getAccelConfig().name
              << " has " << VACs.size() << " VAQs." << std::endl;
#endif
    // Then add the new VAQ to the accelerator
    arax_accel_add_vaccel((*paccel_rr)->arax_accel, vac);
    paccel_rr++;
  }
}
//#define BREAKDOWNS
/*Used from accelThreads to assign a VAQ to an accelthread*/
arax_task_msg_s *ElasticScheduler::selectTask(accelThread *th) {
  auto &VACs = th->getAssignedVACs();

  if (VACs.size() == 0) {
    return 0;
  }
  auto rr_offset = (rr_counter[th] + 1) % VACs.size();
  auto rr_point = VACs.begin() + rr_offset;
  auto itr = rr_point;
  arax_task_msg_s *task = 0;
  for (itr = rr_point; itr != VACs.end(); itr++) {
    task = (arax_task_msg_s *)utils_queue_pop(arax_vaccel_queue(*itr));
    if (task) {
      // Migrate data only if there is a new VA
      if (*itr == vaq4migration) {
#ifdef BREAKDOWNS
        auto start_1 = std::chrono::high_resolution_clock::now();
#endif

        th->migrateFromRemote(task);
        th->migrateToRemote(task);

#ifdef BREAKDOWNS
        auto end_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_milli1 =
            end_1 - start_1;
        char *taskName =
            ((arax_object_s)(((arax_proc_s *)(task->proc))->obj)).name;

        std::cerr << "Task: " << taskName
                  << " sync_to time : " << elapsed_milli1.count() << " ms"
                  << std::endl;
#endif
      }
      goto FOUND_TASK;
    }
  }
  for (itr = VACs.begin(); itr != rr_point; itr++) {
    task = (arax_task_msg_s *)utils_queue_pop(arax_vaccel_queue(*itr));
    if (task) {
      if (*itr == vaq4migration) {
#ifdef BREAKDOWNS
        auto start_1 = std::chrono::high_resolution_clock::now();
#endif

        th->migrateFromRemote(task);
        th->migrateToRemote(task);

#ifdef BREAKDOWNS
        auto end_1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_milli1 =
            end_1 - start_1;
        char *taskName =
            ((arax_object_s)(((arax_proc_s *)(task->proc))->obj)).name;

        std::cerr << "Task: " << taskName
                  << " sync_to time : " << elapsed_milli1.count() << " ms"
                  << std::endl;
#endif
      }
      goto FOUND_TASK;
    }
  }
  return 0;
FOUND_TASK:
  rr_counter[th] = itr - VACs.begin();
  return task;
}
REGISTER_SCHEDULER(ElasticScheduler)
