#include "VacBalancerThread.h"
#include "arax_pipe.h"
#include "core/arax_accel.h"
#include <iostream>

VacBalancerThread ::VacBalancerThread(arax_pipe_s *pipe, Config &conf)
    : run(true),
      conf(conf), std::thread(VacBalancerThread::thread, this, pipe) {}

VacBalancerThread ::~VacBalancerThread() { run = false; }

void VacBalancerThread ::thread(VacBalancerThread *vbt, arax_pipe_s *pipe) {
  do {
    arax_vaccel_s *vac = arax_pipe_get_orphan_vaccel(pipe);
    if (!vac)
      //  XXX HACKKKK!!!!!
      vac = arax_pipe_get_orphan_vaccel(pipe);
    // arax_assert(vac);
    vbt->conf.getSchedulers()[0]->assignVac(vac);
  } while (vbt->run);
}
