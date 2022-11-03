#ifndef OCCUPANCY_CUH
#define OCCUPANCY_CUH

#include <cuda.h>
#include <cupti.h>

typedef struct cupti_eventData_st {
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
} cupti_eventData;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  cupti_eventData *eventData;
  uint64_t eventVal;
} RuntimeApiTrace_t;

void start_event_collection();

void stop_event_collection();

void get_occupancy();

void start_sampling();

void stop_sampling();

#endif
