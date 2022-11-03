#ifndef GPU_ACCELTHREAD
#define GPU_ACCELTHREAD
#include "core/arax_data.h"
#include "cuda_runtime.h"
#include "timers.h"
#include <atomic>
#include <mutex>
#include <pthread.h>
#include <set>
#include <unordered_map>
#include <vector>
class GPUaccelThread;

#include "accelThread.h"
class GPUaccelThread : public accelThread {
public:
  GPUaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
  ~GPUaccelThread();
  virtual void printOccupancy();
  virtual size_t getAvailableSize();
  virtual size_t getTotalSize();
  static cudaStream_t getStream(accelThread *thread);
  static accelThread *getThread(cudaStream_t stream);
  IMPLEMENTS_DEVICE_BASE_OPS()
private:
  int64_t pciId;
  bool host_register;
  cudaStream_t stream;
  static std::unordered_map<accelThread *, cudaStream_t> a2s;
  static std::unordered_map<cudaStream_t, accelThread *> s2a;
};
#endif
