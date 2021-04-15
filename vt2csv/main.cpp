#include <vine_pipe.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

typedef long int Timestamp;

// This is not the unix epoch, but rather
// a time just before the metrics recording
Timestamp epoch;

std::vector<std::string> metric_names;
std::vector<const size_t *> metric_values;
volatile bool run = true;
static inline Timestamp get_now()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count() - epoch;
}

void start_recording(std::ostream & os)
{
    for (auto metric : metric_names)
        os << metric << ", ";
    os << std::endl;

    epoch = get_now();
    std::cerr << "Waiting for virtual accelerator";
    while (*metric_values[VINE_TYPE_VIRT_ACCEL] == 0) {
        auto now = get_now();
        if (now > 200000) {
            std::cerr << '.';
            epoch = 0;
            epoch = get_now();
        }
    }
    std::cerr << std::endl;

    epoch = 0;
    epoch = get_now();

    while (run) {
        std::ostringstream ss;
        auto start = get_now();
        for (auto metric : metric_values)
            ss << *metric << ", ";
        auto end = get_now();
        ss << std::endl;
        os << (start + end) / 2 << ", "
           << (end - start) << ", "
           << ss.str();
    }
}

int main(int argc, char *argv[])
{
    vine_pipe_s *vpipe = vine_talk_init();

    metric_names.push_back("Time");
    metric_names.push_back("Span");

    const char *typestr[VINE_TYPE_COUNT] =
    {
        "Phys Accel",
        "Virt Accel",
        "Vine Procs",
        "Vine Datas",
        "Vine Tasks"
    };

    for (int type = 0; type < VINE_TYPE_COUNT; type++) {
        metric_names.push_back(typestr[type]);
        utils_list_s *list = vine_object_list_lock(&(vpipe->objs), (vine_object_type_e) type);
        metric_values.push_back(&(list->length));
        vine_object_list_unlock(&(vpipe->objs), (vine_object_type_e) type);
    }

    if (argc == 2) {
        std::ofstream fout(argv[1]);
        if (!fout) {
            std::cerr << "Could not open " << argv[1] << std::endl;
            return 1;
        }
        start_recording(fout);
    } else {
        start_recording(std::cerr);
    }

    vine_talk_exit();
    return 0;
} // main
