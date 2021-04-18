#include <vine_pipe.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Timestamp.h"
#include "Sample.h"
#include "Metric.h"
#include "Phase.h"

size_t* getVineObjectCounter(vine_pipe_s *vpipe, vine_object_type_e type)
{
    size_t *ret;
    utils_list_s *list = vine_object_list_lock(&(vpipe->objs), (vine_object_type_e) type);

    ret = &(list->length);
    vine_object_list_unlock(&(vpipe->objs), (vine_object_type_e) type);
    return ret;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage:\n\t" << argv[0] << " <output_file>\n\n";
        return -1;
    }

    vine_pipe_s *vpipe = vine_talk_init();

    size_t *vaqs = getVineObjectCounter(vpipe, VINE_TYPE_VIRT_ACCEL);

    std::function<bool()> start_cond = [vaqs](){
          return *vaqs != 0;
      };

    Trace trace(argv[1], start_cond);

    const char *typestr[VINE_TYPE_COUNT] =
    {
        "Phys Accel",
        "Virt Accel",
        "Vine Procs",
        "Vine Datas",
        "Vine Tasks"
    };

    vine_object_type_e types_to_plot[] =
    {
        VINE_TYPE_VIRT_ACCEL,
        VINE_TYPE_DATA,
        VINE_TYPE_TASK
    };

    for (int ttp = 0; ttp < 3; ttp++)
        trace.addMetric(typestr[types_to_plot[ttp]], getVineObjectCounter(vpipe, types_to_plot[ttp]));

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

    vine_talk_exit();
    return 0;
} // main
