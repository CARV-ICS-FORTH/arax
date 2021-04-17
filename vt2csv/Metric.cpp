#include "Metric.h"
#include "Timestamp.h"
#include "Pallete.h"
#include <atomic>
#include <unistd.h>
#include <sched.h>
#include <signal.h>

SampleList samples;
std::vector<std::string> metric_names;
std::vector<const size_t *> metric_values;
std::atomic<bool> run;

const size_t *start_condition;

void add_metric(std::string name, const size_t *value)
{
    metric_names.push_back(name);
    metric_values.push_back(value);
}

void prep_recording(const size_t *cond)
{
    start_condition = cond;
    run = true;
    signal(SIGINT, [](int sig){
        run = false;
    });
}

void start_recording()
{
    reset_epoch();

    std::cerr << "Waiting for virtual accelerator";
    while (*start_condition == 0 && run) {
        auto now = get_now();
        if (now > 200000) {
            std::cerr << '.';
            reset_epoch();
        }
    }
    std::cerr << std::endl;

    std::cerr << "Recording: ";

    reset_epoch();

    while (run) {
        samples.emplace_front(metric_values);
    }

    std::cerr << "Done" << std::endl;
}

SampleList & get_samples()
{
    return samples;
}

const char *html[100] = {
    "<!doctype html><html><head><meta charset='utf-8'><title>",
    "</title><meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<link rel='stylesheet' href='https://leeoniya.github.io/uPlot/dist/uPlot.min.css'>"
    "</head><body><script src='https://leeoniya.github.io/uPlot/dist/uPlot.iife.min.js'></script>"
    "<style>body{margin:0;background:#555}</style>"
    "<script>function makeChart(data){let opts={title:'",
    "',width:1900,height:800,scales:{x:{time:false,}},series:["
    "{label:'Time(us)'}",
    "]};let uplot=new uPlot(opts,data,document.body);}let data = ",
    ";makeChart(data);</script></body></html>"
};

std::ostream & operator << (std::ostream & os, const SampleList & samples)
{
    os << "[\n";
    os << "[";

    std::string sep = "";

    for (auto & s : samples) {
        os << sep << s.start;
        sep = ",";
    }
    os << "]\n";

    for (int c = 0; c < samples.front().values.size(); c++) {
        os << ",[";
        std::string sep = "";
        for (auto & s : samples) {
            os << sep << s.values[c];
            sep = ",";
        }
        os << "]\n";
    }

    os << "]";
    return os;
}

void write_metrics(std::ostream & os, std::string title)
{
    os << html[0] << title;
    os << html[1] << title;
    os << html[2];

    for (int n = 0; n < metric_names.size(); n++) {
        os << ",{label:'" << metric_names[n] << "',stroke:'#" << Pallete::get(n, 12) << "',width:1/devicePixelRatio,}";
    }

    os << html[3] << samples;
    os << html[4];
}
