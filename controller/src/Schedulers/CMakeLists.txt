set(
  SCHEDULERS_CPP
  #       BaselineCheckpointScheduler.cpp
  #       FlexyRRScheduler.cpp
  #       LoadBalanceScheduler.cpp
  #       MemoryUsageCheckpointScheduler.cpp
  #       TimeBasedCheckpointScheduler.cpp
  RoundRobinScheduler.cpp
  HeterogeneousScheduler.cpp
  ElasticScheduler.cpp
  #       VoidCheckpointScheduler.cpp
  #       WeightedRoundRobinScheduler.cpp
)

add_library(schedulers OBJECT ${SCHEDULERS_CPP})
