#ifndef CPU_ACCELTHREAD
#define CPU_ACCELTHREAD
#include "timers.h"
#include <pthread.h>
#include <vector>
class CPUaccelThread;

#include "accelThread.h"

class CPUaccelThread : public accelThread {
public:
    CPUaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
    bool checkpoint(arax_task_msg_s *arax_task);
    virtual size_t getAvailableSize();
    virtual size_t getTotalSize();
    ~CPUaccelThread();
    IMPLEMENTS_DEVICE_BASE_OPS();
};
#endif // ifndef CPU_ACCELTHREAD
