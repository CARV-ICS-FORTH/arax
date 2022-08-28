#include <arax_pipe.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Timestamp.h"
#include "Sample.h"
#include "Metric.h"
#include "Phase.h"

std::size_t* getAraxObjectCounter(arax_pipe_s *vpipe, arax_object_type_e type)
{
    std::size_t *ret;
    utils_list_s *list = arax_object_list_lock(&(vpipe->objs), (arax_object_type_e) type);

    ret = &(list->length);
    arax_object_list_unlock(&(vpipe->objs), (arax_object_type_e) type);
    return ret;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage:\n\t" << argv[0] << " <output_file>\n\n";
        return -1;
    }

    arax_pipe_s *vpipe = arax_init();

    std::size_t *vaqs = getAraxObjectCounter(vpipe, ARAX_TYPE_VIRT_ACCEL);

    std::function<bool()> start_cond = [vaqs](){
          return *vaqs != 0;
      };

    Trace trace(argv[1], start_cond);

    const char *typestr[ARAX_TYPE_COUNT] =
    {
        "Phys Accel",
        "Virt Accel",
        "Arax Procs",
        "Arax Datas",
        "Arax Tasks"
    };

    arax_object_type_e types_to_plot[] =
    {
        ARAX_TYPE_VIRT_ACCEL,
        ARAX_TYPE_DATA,
        ARAX_TYPE_TASK
    };

    utils_list_s *list = arax_object_list_lock(&(vpipe->objs), (arax_object_type_e) ARAX_TYPE_PHYS_ACCEL);

    utils_list_node_s *itr;

    utils_list_for_each(*list, itr)
    {
        arax_accel_s *obj = (arax_accel_s *) (itr->owner);

        trace.addMetric("Tasks " + std::string(obj->obj.name), (size_t *) (&(obj)->tasks) );
        trace.addMetric("VAQs " + std::string(obj->obj.name), (size_t *) (&((obj)->vaccels.length)) );
    }

    arax_object_list_unlock(&(vpipe->objs), (arax_object_type_e) ARAX_TYPE_PHYS_ACCEL);

    for (int ttp = 0; ttp < 3; ttp++)
        trace.addMetric(typestr[types_to_plot[ttp]], getAraxObjectCounter(vpipe, types_to_plot[ttp]));

    for (int metric = 0; metric < ARAX_KV_CAP; metric++) {
        auto & m = vpipe->metrics_kv.kv[metric];
        if (m.key) {
            std::cerr << "Added extra metric " << ((char *) m.key) << std::endl;
            trace.addMetric((const char *) m.key, (size_t *) m.value);
        }
    }

    trace.start();

    SampleList & samples = trace.getSamples();

    if (samples.empty()) {
        std::cerr << "\nNo samples were recorded!\n";
        return 0;
    }

    // Remove idle/constant samples from end of run
    {
        Phase p("Trimming idle time at end");
        TrimIdleEnd trim(samples);

        trace.removeSamples(trim);
    }

    // Remove sequences of same samples
    {
        Phase p("Removing duplicate samples");
        SampleList::iterator prev  = samples.begin();
        SampleList:: iterator curr = prev;

        curr++;
        SampleList:: iterator next = curr;

        next++;

        while (next != samples.end()) {
            if (*prev == *curr && *curr == *next) { // Found 3 same samples, remove the middle/current one
                samples.erase_after(prev);
                curr = next;
                next++;
            } else { // Not a sequence, move all forward
                prev++;
                curr++;
                next++;
            }
        }
    }

    // Revese
    {
        Phase p("Reversing list");
        samples.reverse();
    }

    // Write results
    {
        std::string title = argv[1];
        Phase p("Writing results in " + title);

        title += ".html";
        std::ofstream ofs(title.c_str());

        ofs << trace;
        ofs.close();
    }

    arax_exit();
    return 0;
} // main
