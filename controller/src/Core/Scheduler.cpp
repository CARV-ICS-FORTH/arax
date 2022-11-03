#include "Scheduler.h"
#include <iostream>
Scheduler::Scheduler(picojson::object args) {}

void Scheduler ::setGroup(GroupConfig *group) { this->group = group; }
void Scheduler ::setConfig(Config *config) { this->config = config; }

Scheduler::~Scheduler() {}

void Scheduler::postTaskExecution(accelThread *th, arax_task_msg_s *task) {}

void Scheduler::accelThreadSetup(accelThread *th) {
  // No accelThreadSetup for default scheduler.
}

Factory<Scheduler, picojson::object> schedulerFactory;
