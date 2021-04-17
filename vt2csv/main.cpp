#include <vine_pipe.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Timestamp.h"
#include "Sample.h"
#include "Metric.h"

class TrimIdleEnd
{
private:
    Sample last;
    bool trim;
public:
    TrimIdleEnd(const SampleList & list)
        : last(list.front()), trim(true)
    { }

    bool operator () (const Sample & s)
    {
        trim = trim && (s == last);
        return trim;
    };
};

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage:\n\t" << argv[0] << " <output_file>\n\n";
        return -1;
    }

    vine_pipe_s *vpipe = vine_talk_init();

    const char *typestr[VINE_TYPE_COUNT] =
    {
        "Phys Accel",
        "Virt Accel",
        "Vine Procs",
        "Vine Datas",
        "Vine Tasks"
    };

    for (int type = 0; type < VINE_TYPE_COUNT; type++) {
        utils_list_s *list = vine_object_list_lock(&(vpipe->objs), (vine_object_type_e) type);
        add_metric(typestr[type], &(list->length));

        // Use number of vaqs as a start condition
        if (type == VINE_TYPE_VIRT_ACCEL)
            prep_recording(&(list->length));

        vine_object_list_unlock(&(vpipe->objs), (vine_object_type_e) type);
    }

    start_recording();

    SampleList & samples = get_samples();

    if (samples.empty()) {
        std::cerr << "\nNo samples were recorded!\n";
        return 0;
    }

    // Remove idle/constant samples from end of run
    std::cerr << "Trimming idle time at end: ";
    TrimIdleEnd trim(samples);

    samples.remove_if(trim);
    std::cerr << "Done" << std::endl;

    // Remove sequences of same samples
    std::cerr << "Removing duplicate samples: ";
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

    std::cerr << "Done" << std::endl;

    // Revese
    std::cerr << "Reversing list: ";
    samples.reverse();
    std::cerr << "Done" << std::endl;
    std::string title = argv[1];

    title += ".html";
    std::ofstream ofs(title.c_str());

    std::cerr << "Writing results in " << title << ": ";
    write_metrics(ofs, argv[1]);
    ofs.close();
    std::cerr << "Done" << std::endl;

    vine_talk_exit();
    return 0;
} // main
