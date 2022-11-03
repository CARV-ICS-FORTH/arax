#ifndef HIP_ACCELTHREAD
#define HIP_ACCELTHREAD
#include "core/arax_data.h"
#include "hip/hip_runtime.h"
#include "timers.h"
#include <atomic>
#include <mutex>
#include <pthread.h>
#include <set>
#include <unordered_map>
#include <vector>
class HIPaccelThread;

#include "accelThread.h"
class HIPaccelThread : public accelThread {
public:
  HIPaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
  ~HIPaccelThread();
  virtual void printOccupancy();
  virtual size_t getAvailableSize();
  virtual size_t getTotalSize();
  static hipStream_t getStream(accelThread *thread);
  static accelThread *getThread(hipStream_t stream);
  IMPLEMENTS_DEVICE_BASE_OPS();

private:
  int64_t pciId;
  hipStream_t stream;
  static std::unordered_map<accelThread *, hipStream_t> a2s;
  static std::unordered_map<hipStream_t, accelThread *> s2a;
  std::set<std::string> ptx_set;
};
#endif
