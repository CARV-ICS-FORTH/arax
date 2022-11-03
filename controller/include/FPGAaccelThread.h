#ifndef FPGA_ACCELTHREAD
#define FPGA_ACCELTHREAD
#include "timers.h"
#include <pthread.h>
#include <vector>
class FPGAaccelThread;

#include "accelThread.h"

class FPGAaccelThread : public accelThread {
public:
    FPGAaccelThread(arax_pipe_s *v_pipe, AccelConfig &conf);
    virtual bool checkpoint(arax_task_msg_s *arax_task);
    ~FPGAaccelThread();
    IMPLEMENTS_DEVICE_BASE_OPS();
};
#endif // ifndef FPGA_ACCELTHREAD
