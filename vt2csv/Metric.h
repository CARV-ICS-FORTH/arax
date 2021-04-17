#ifndef VT2CSV_METRIC
#define VT2CSV_METRIC
#include <iostream>
#include "Sample.h"

void add_metric(std::string name, const size_t *value);

void prep_recording(const size_t *cond);

void start_recording();

SampleList & get_samples();

void write_metrics(std::ostream & os, std::string title);

#endif // ifndef VT2CSV_METRIC
