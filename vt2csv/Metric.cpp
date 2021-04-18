#include "Metric.h"
#include "Timestamp.h"
#include "Pallete.h"
#include "Phase.h"
#include <unistd.h>
#include <sched.h>
#include <signal.h>

std::atomic<bool> Trace::run;

Trace :: Trace(std::string name, std::function<bool()> & start_cond)
    : name(name), start_cond(start_cond)
{
    run = true;
    signal(SIGINT, [](int sig){
        std::cerr << "\b\b";
        run = false;
    });
}

void Trace :: addMetric(std::string name, const size_t *value)
{
    metric_names.push_back(name);
    metric_values.push_back(value);
}

void Trace :: start()
{
    reset_epoch();

    {
        Phase p("Waiting for start condition");
        while (!start_cond() && run) {
            auto now = get_now();
            if (now > 200000) {
                reset_epoch();
            }
        }
    }

    {
        Phase p("Recording");

        reset_epoch();

        while (run) {
            samples.emplace_front(metric_values);
        }
    }
}

SampleList & Trace :: getSamples()
{
    return samples;
}

void Trace :: removeSamples(std::function<bool(const Sample & s)> fn)
{
    samples.remove_if(fn);
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

std::ostream & operator << (std::ostream & os, const Trace & t)
{
    os << html[0] << t.name;
    os << html[1] << t.name;
    os << html[2];

    for (int n = 0; n < t.metric_names.size(); n++) {
        os << ",{label:'" << t.metric_names[n] << "',stroke:'#" <<
            Pallete::get(n, 12) << "',width:1/devicePixelRatio,}";
    }

    os << html[3] << t.samples;
    os << html[4];
    return os;
}
