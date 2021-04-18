#ifndef VT2CSV_METRIC
#define VT2CSV_METRIC
#include <iostream>
#include "Sample.h"
#include <atomic>
#include <functional>


class Trace
{
public:
    friend std::ostream & operator << (std::ostream & os, const Trace & t);
    Trace(std::string name, std::function<bool()> & start_cond);
    void addMetric(std::string name, const size_t *value);
    void prepare();
    void start();
    SampleList & getSamples();
    void removeSamples(std::function<bool(const Sample & s)> fn);
private:
    std::string name;
    std::function<bool()> & start_cond;
    std::vector<std::string> metric_names;
    std::vector<const size_t *> metric_values;
    SampleList samples;
    static std::atomic<bool> run;
};


void add_metric(std::string name, const size_t *value);

void prep_recording(const size_t *cond);

void start_recording();

SampleList & get_samples();

void write_metrics(std::ostream & os, std::string title);

#endif // ifndef VT2CSV_METRIC
