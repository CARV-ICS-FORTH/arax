#ifndef VAC_BALANCER_THREAD_HEADER
#define VAC_BALANCER_THREAD_HEADER
#include "Config.h"
#include "arax_pipe.h"
#include <thread>

class VacBalancerThread : public std::thread {
public:
    VacBalancerThread(arax_pipe_s *pipe, Config &conf);
    ~VacBalancerThread();

private:
    Config &conf;
    static void thread(VacBalancerThread *vbt, arax_pipe_s *pipe);
    bool run;
};

#endif // ifndef VAC_BALANCER_THREAD_HEADER
