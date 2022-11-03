#ifndef FREE_THREAD_HEADER
#define FREE_THREAD_HEADER
#include "Config.h"
#include "arax_pipe.h"
#include <thread>

class FreeThread : public std::thread {
public:
  FreeThread(arax_pipe_s *pipe, Config &conf);
  ~FreeThread();
  void terminate();

private:
  Config &conf;
  static void thread(FreeThread *vbt, arax_pipe_s *pipe);
  bool run;
};

#endif
