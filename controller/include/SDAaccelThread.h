#ifndef SDA_ACCELTHREAD
#define SDA_ACCELTHREAD
#include "timers.h"
#include "xcl.h"
#include <map>
#include <pthread.h>
class SDAaccelThread;

#include "accelThread.h"

class SDAaccelThread : public accelThread {
public:
  SDAaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
  ~SDAaccelThread();
  virtual bool
  acceleratorInit(); /* Function that initializes a SDA accelerator */
  virtual void
  acceleratorRelease(); /* Function that resets a SDA accelerator */
  virtual void executeOperation(
      void *functor,
      arax_task_msg_s
          *task); /*Executes an operation (syncTo, SyncFrom, free, kernel)*/
private:
  CpuSet prof_thread_cpus; /* Pin Xilinx profiling thread*/
  std::string vendor;
  std::string dev_addr_str;
  std::string xclbin;                       /** xcl bin file */
  std::map<std::string, cl_kernel> kernels; /** Available kernels */
  xcl_world world;
};
#endif
