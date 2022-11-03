#include "accelThread.h"
#include "Csv.h"
#include "Services.h"
#include "Utilities.h"
#include "arax_pipe.h"
#include "core/arax_data.h"
#include "core/arax_ptr.h"
#include "definesEnable.h"
#include "utils/timer.h"
#include <atomic>
#include <chrono>
#include <ctime>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <pthread.h>
#include <string.h>
//#define DEBUG_MIGRATION
#include <vector>
#ifdef DEBUG_MIGRATION
std::chrono::high_resolution_clock::time_point s1_mig;
std::chrono::high_resolution_clock::time_point s2_mig;
std::chrono::high_resolution_clock::time_point s3_mig;
std::chrono::high_resolution_clock::time_point e1_mig;
std::chrono::high_resolution_clock::time_point e2_mig;
std::chrono::high_resolution_clock::time_point e3_mig;
#endif

#define USERF_TASK 1
//#define MEASSURE_TIME
using namespace ::std;
unordered_map<arax_accel *, accelThread *> accelThread::VAQPhys2accelThread;
std::mutex print_lock;
std::mutex thread_prep;
std::mutex accelThreadSetup;

void *workFunc(void *thread);

bool isUserfacingTask(arax_accel *accel) {
  bool isUser =
      arax_vaccel_get_job_priority((arax_vaccel_s *)accel) == USERF_TASK;
  return isUser;
}

// Constructor
accelThread::accelThread(arax_pipe_s *v_pipe, AccelConfig &accelConf)
    : readyToServe(false), served_tasks(0),
      enable_mtx(PTHREAD_MUTEX_INITIALIZER), accelConf(accelConf),
      v_pipe(v_pipe) {
  revision = 0;
  stopExec = 0;
  migrations = 0;
#ifdef FREE_THREAD
  void *mem = arch_alloc_allocate(&(v_pipe->allocator), sizeof(utils_queue_s));
  freeTaskQueue = utils_queue_init(mem);
#endif
  // add operations to ignore for checkpointing
  this->operations2Ignore.insert("init_phys");
  this->operations2Ignore.insert("alloc");
  this->operations2Ignore.insert("alloc_data");
  this->operations2Ignore.insert("free");
  this->operations2Ignore.insert("memset");
  this->operations2Ignore.insert("syncFrom");
  this->operations2Ignore.insert("syncTo");
}
#ifdef FREE_THREAD
/*returns the queue with the tasks that need ref_dec*/
utils_queue_s *accelThread::getFreeTaskQueue() { return freeTaskQueue; }
#endif

bool accelThread::isReadyToServe() { return readyToServe; }

void accelThread::start() {
  pthread_create(&pthread, NULL, workFunc, this);
  pthread_setaffinity_np(pthread, sizeof(cpu_set_t),
                         accelConf.affinity.getSet());
}

arax_accel_type_e accelThread::getAccelType() {
  return accelConf.arax_accel->type;
}

/*Returns all VAQs of the system (with ANY type)*/
const std::vector<arax_vaccel_s *> &accelThread::getAssignedVACs() {
  return assignedVACs;
}

/*Returns the current configuration*/
AccelConfig &accelThread::getAccelConfig() { return accelConf; }

/*Get total size */
size_t accelThread::getAvailableSize() {
  throw std::runtime_error(__func__ +
                           (" not implemented for " + accelConf.type_str));
  return 0;
}
int accelThread::getNumMigrations() { return this->migrations; }
void accelThread::incMigrations() { this->migrations++; }

/*Get total size */
size_t accelThread::getTotalSize() {
  throw std::runtime_error(__func__ +
                           (" not implemented for " + accelConf.type_str));
  return 0;
}

AccelConfig::JobPreference accelThread::getJobPreference() {
  return accelConf.job_preference;
}

AccelConfig::JobPreference accelThread::getInitialJobPreference() {
  return accelConf.initial_preference;
}

void accelThread::setJobPreference(AccelConfig::JobPreference jpref) {
  accelConf.job_preference = jpref;
  if (jpref == AccelConfig::NoJob)
    disable();
  else
    enable();
}

void accelThread::enable() { pthread_mutex_unlock(&enable_mtx); }

void accelThread::disable() { pthread_mutex_trylock(&enable_mtx); }

/*Stops the execution*/
void accelThread::terminate() {
  stopExec = 1;
  /*Add a fake task to sleep the current thread */
  arax_accel_add_task(accelConf.arax_accel);
  setJobPreference(AccelConfig::AnyJob);
}

void accelThread::joinThreads() { pthread_join(pthread, 0); }

arax_pipe_s *accelThread::getPipe() { return v_pipe; }

void accelThread::printOccupancy() { /*Used for GPUs only*/
}

size_t accelThread::getServedTasks() { return served_tasks; }

// returns the VAQ->phys of an accelThread
accelThread *accelThread::getVAQPhys2accelThread(arax_accel *phys) {
  if (phys == 0) {
    return 0;
  }
  accelThread *ret = VAQPhys2accelThread.at(phys);
  return ret;
}
// add a pair of physicall accelerator and accelThread
void accelThread::addVAQPhys2accelThread(accelThread *th, arax_accel *phys) {
  VAQPhys2accelThread.insert({phys, th});
  /*
  for (auto const &pair:VAQPhys2accelThread) {
      std::cout << "{" << pair.first << ": " << pair.second << "}\n";
  }
  */
}

/*Destructor*/
accelThread::~accelThread() {}

void printAcceleratorType(GroupConfig *group) {
  map<int, int> acceleratorCount;
  for (auto accel : group->getAccelerators()) {
    acceleratorCount[accel.second->accelthread->getJobPreference()]++;
  }
  for (int i = 0; i < 4; i++) {
    cout << " " << acceleratorCount[i];
  }
}
void print(string str, arax_throttle_s &th) {
  size_t a = arax_throttle_get_available_size(&th);
  size_t t = arax_throttle_get_total_size(&th);
  cout << str << " " << (t - a) / (1024 * 1024) << " mb\n";
}
/*Migrates the data of a task to REMOTE*/
void accelThread::migrateToRemote(arax_task_msg_s *task) {
#ifdef MIGRATION
  arax_data_s *data;
  for (int arg = 0; arg < task->in_count + task->out_count; arg++) {

#ifdef DEBUG_MIGRATION
    char *taskName = ((arax_object_s)(((arax_proc_s *)(task->proc))->obj)).name;

    std::cerr << __func__ << " Task[ " << taskName << "] from thread "
              << this->getAccelConfig().name << std::endl;
    s1_mig = std::chrono::high_resolution_clock::now();
#endif

    data = (arax_data_s *)task->io[arg];
    accelThread *tmp = getVAQPhys2accelThread(data->phys);
    // The thread that had the data is different than the thread that we
    // are going to run the next task. So migrate data.
    if (tmp != this) {
      // allocate memory to the accelerator + dec throttling counter

      this->alloc_remote(data);

      // transfer data to the new accelerator
      this->sync_to_remote(data);

      // This the first time we see a data, so there is no phys.
    } else if (tmp == 0) {
      std::cerr << __func__ << " TMP==0 for data: " << data << std::endl;
      this->alloc_remote(data);
    } else if (tmp == this) { // I have already the data
      continue;
    } else { // Not known case abort!!
      abort();
    }
  }
#ifdef DEBUG_MIGRATION
  e1_mig = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> mig_milli = e1_mig - s1_mig;

  std::cerr << __func__ << " Task[ " << taskName << "] from thread "
            << this->getAccelConfig().name
            << " data size: " << arax_data_size(data)
            << " elapsed time: " << mig_milli.count() << "ms" << std::endl;
#endif

#endif
}

/*Migrates the data of a task to SHM*/
void accelThread::migrateFromRemote(arax_task_msg_s *task) {
#ifdef MIGRATION
  arax_data_s *data;
  for (int arg = 0; arg < task->in_count + task->out_count; arg++) {
#ifdef DEBUG_MIGRATION
    char *taskName = ((arax_object_s)(((arax_proc_s *)(task->proc))->obj)).name;
    std::cerr << __func__ << " Task [" << taskName << "] from thread "
              << this->getAccelConfig().name << std::endl;
    s2_mig = std::chrono::high_resolution_clock::now();
#endif

    data = (arax_data_s *)task->io[arg];
    accelThread *oldTh = getVAQPhys2accelThread(data->phys);
    if (oldTh == 0) { // data do not have phys (data->phys = 0).
      // It is acceptable only for outputs
      continue;
    } else if (oldTh != this) { // Data are in another accelerator
      // copy data from remote of the other thread
      oldTh->sync_from_remote(data);
      // free old data
      oldTh->free_remote(data);

#ifdef DEBUG_MIGRATION
      e2_mig = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> mig2_milli = e2_mig - s2_mig;

      std::cerr << __func__ << " Task[ " << taskName << "] from thread "
                << this->getAccelConfig().name
                << " data size: " << arax_data_size(data)
                << " sync_from_and_free: " << mig3_milli.count() << "ms"
                << std::endl;
#endif

    } else if (oldTh == this) { // I have the requested data
      continue;
    } else { // not known case ABORT
      abort();
    }
  }
#endif
}

extern arax_pipe_s *vpipe_s;
/*Function that handles the Virtual accelerator queues */
void *workFunc(void *thread) {
#ifdef MEASSURE_TIME
  chrono::time_point<chrono::system_clock> start_part[3];
  chrono::time_point<chrono::system_clock> end_part[3];
  chrono::duration<double, micro> duration[3];
  vector<string> fileNames;
  fileNames.push_back("part1.txt");
  fileNames.push_back("part2.txt");
  fileNames.push_back("part3.txt");
  vector<double> elapsed[3];
  for (int j = 0; j < 3; j++)
    elapsed[j].reserve(1000000);
#endif

  accelThread *th = (accelThread *)thread;
  /*create an arax task pointer*/
  arax_task_msg_s *arax_task;

  /*The scheduling policy to be used*/
  Scheduler *selectedScheduler;
  selectedScheduler = th->accelConf.group->getScheduler();

  accelThreadSetup.lock();
  selectedScheduler->accelThreadSetup(th);
  accelThreadSetup.unlock();

  {
    set_thread_name(th->getAccelConfig().name + "#Rt");
    std::lock_guard<std::mutex> lg(print_lock);
    std::cerr << th->getAccelConfig().name << "@" << (void *)th << " :: "
              << "Start initialization" << std::endl;
    /*Initiliaze the accelerator*/
    if (!th->acceleratorInit()) {
      std::cerr << th->getAccelConfig().name << " :: " << ESC_CHR(ANSI_RED)
                << th->getAccelConfig().name << " initialization failed"
                << ESC_CHR(ANSI_RST) << std::endl;
      return 0;
    }
    string msg;
    th->initialAvailableSize = th->getAvailableSize();
    th->getAccelConfig().arax_accel =
        arax_accel_init(vpipe_s, (char *)(th->getAccelConfig().name.c_str()),
                        th->getAccelConfig().type, th->initialAvailableSize,
                        th->getTotalSize());
    if (th->getAccelConfig().arax_accel == 0) {
      msg = "Failed to perform initialization";
      cerr << "While spawning" << th->getAccelConfig().arax_accel << msg
           << " for " << th->getAccelConfig().name << endl;
    }

    cout << "Done." << endl;

    std::cerr << th->getAccelConfig().name << " :: " << ESC_CHR(ANSI_GREEN)
              << "Initialization successful" << ESC_CHR(ANSI_RST) << std::endl;
    std::cerr << std::endl
              << th->getAccelConfig().name << " :: " << ESC_CHR(ANSI_GREEN)
              << "Ready" << ESC_CHR(ANSI_RST) << std::endl;
  }

  set_thread_name(th->getAccelConfig().name + "#At");
  /*Type of the accelerator thread*/
  arax_accel_type_e accelThreadType =
      (arax_accel_type_e)(th->accelConf).arax_accel->type;

  th->readyToServe = true;
  thread_prep.lock();
  th->addVAQPhys2accelThread(th, th->accelConf.arax_accel);
  thread_prep.unlock();

  /*Iterate until ctrl+c is pressed*/
  while (!th->stopExec) {
#ifdef MEASSURE_TIME
    start_part[0] = chrono::system_clock::now();
#endif

    if (th->accelConf.job_preference == AccelConfig::NoJob)
      pthread_mutex_lock(&(th->enable_mtx));

    th->updateVirtAccels();
    /*iterate in the Virtual accelerator vector, which contains all the VAQs*/
    arax_task = selectedScheduler->selectTask(th);
    /*If there is a VALID arax_task_msg */
    if (arax_ptr_valid(arax_task)) {
      th->served_tasks++;
      /*Name of task*/
      if (!arax_task->proc) {
        std::cerr << "Task with NULL proc recieved!\n";
        continue;
      }
      char *taskName =
          ((arax_object_s)(((arax_proc_s *)(arax_task->proc))->obj)).name;

      /*kernel of the selected task*/
      arax_proc_s *proc;

      /*Get the currently executed task*/
      th->runningTask = arax_task;

      proc = (arax_proc_s *)arax_task->proc;
      arax_assert(proc);
      arax_assert(arax_task->accel);

      arax_object_ref_inc(&(arax_task->obj));

#ifdef MEASSURE_TIME
      end_part[0] = chrono::system_clock::now();
      duration[0] = end_part[0] - start_part[0];
      elapsed[0].push_back(duration[0].count());
      start_part[1] = chrono::system_clock::now();
#endif
      arax_data_s *data;
      int inputs_num = arax_task->in_count;
      int outputs_num = arax_task->in_count + arax_task->out_count;
      for (int arg = inputs_num; arg < outputs_num; arg++) {
        data = (arax_data_s *)arax_task->io[arg];
        th->alloc_remote(data);
      }
      th->executeOperation(arax_proc_get_functor(proc, accelThreadType),
                           arax_task);
#ifdef DEBUG
      std::cerr << "Task name:"
                << "\033[1;31m" << taskName << "\033[0m"
                << " executed from thread: " << th->getAccelConfig().name
                << std::endl;
#endif
#ifdef MEASSURE_TIME
      end_part[1] = chrono::system_clock::now();
      duration[1] = end_part[1] - start_part[1];
      elapsed[1].push_back(duration[1].count());
      start_part[2] = chrono::system_clock::now();
#endif
      // Get the name of the last operation
      // utils_timer_set(arax_task->stats.task_duration_without_issue,
      // stop);

      selectedScheduler->postTaskExecution(th, arax_task);

      /*there is no task running in the GPU*/
      th->runningTask = 0;

#ifdef FREE_THREAD
      utils_queue_push(th->getFreeTaskQueue(), arax_task);
#else
      arax_object_ref_dec(&(arax_task->obj));
#endif

#ifdef MEASSURE_TIME
      end_part[2] = chrono::system_clock::now();
      duration[2] = end_part[2] - start_part[2];
      elapsed[2].push_back(duration[2].count());
#endif
    }
  }

  {
    std::lock_guard<std::mutex> lg(print_lock);
    std::cerr << th->getAccelConfig().name << ESC_CHR(ANSI_GREEN)
              << " :: Releasing" << ESC_CHR(ANSI_RST) << std::endl;

    /*Release the accelerator*/
    arax_throttle_size_inc(&(th->getAccelConfig().arax_accel->throttle),
                           th->getTotalSize() -
                               th->initialAvailableSize); // Release initial
    th->acceleratorRelease();
    std::cerr << th->getAccelConfig().name << " :: " << ESC_CHR(ANSI_GREEN)
              << "Released" << ESC_CHR(ANSI_RST) << std::endl;
    std::cerr << "From " << th->getAccelConfig().name
              << "to other Migrations: " << th->getNumMigrations() << std::endl;
#ifdef MEASSURE_TIME
    int count = 0;
    std::ofstream part[3];
    for (int j = 0; j < 3; j++) {
      part[j].open(fileNames[j], std::ios_base::app);
      for (auto i = elapsed[j].begin(); i != elapsed[j].end(); ++i) {
        part[j] << count << ":" << *i << std::endl;
        count++;
      }
      count = 0;
    }
#endif
  }
  return 0;
}

/*Search for new VAQs with ANY queues*/
void accelThread::updateVirtAccels() {
  arax_accel_wait_for_task(getAccelConfig().arax_accel);
  arax_vaccel_s **allVirtualAccels;
  size_t numOfConcurrentVAQs = arax_accel_get_assigned_vaccels(
      getAccelConfig().arax_accel, &allVirtualAccels);
  assignedVACs.resize(numOfConcurrentVAQs);
  std::copy(allVirtualAccels, allVirtualAccels + numOfConcurrentVAQs,
            assignedVACs.begin());
  //    std::cerr<<"["<<__func__<<"] Thread:"<<this->getAccelConfig().name<<"
  //    assigned VAQS: "
  //        <<assignedVACs.size()<<std::endl;
}

Factory<accelThread, arax_pipe_s *, AccelConfig &> threadFactory;
