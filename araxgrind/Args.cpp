#include "Args.h"
#include <ostream>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sstream>
#include <conf.h>

struct Flag
{
    std::string sflag;
    std::string lflag;
    bool        & value;
    std::string description;
};

bool operator == (const Flag & f, std::string arg)
{
    if (std::count(arg.begin(), arg.end(), '-') == 1) { // Single character flag
        return std::count(arg.begin(), arg.end(), f.sflag[1]);
    }
    if (f.lflag == arg)
        return true;

    return false;
}

std::string exe;
bool help;
bool all;
bool ptr;
bool refresh;
bool inspect;
#ifdef ARAX_DATA_TRACK
bool track;
#endif
std::set<void *> inspect_set;

std::vector<Flag> flags =
{
    { "-h", "--help",    help,    "Show this help message."     },
    { "-a", "--all",     all,     "Show all types of leaks."    },
    { "-p", "--ptr",     ptr,     "Show pointers for leaks."    },
    { "-i", "--inspect", inspect, "Inspect passed pointer."     },
    #ifdef ARAX_DATA_TRACK
    { "-t", "--track",   track,   "Print data tracking string." },
    #endif
    { "-r", "--refresh", refresh, "Refresh output every 250ms." }
};

bool parseArgs(std::ostream & os, int argc, char *argv[])
{
    exe = argv[0];

    for (int arg = 1; arg < argc; arg++) {
        int matched = flags.size();
        for (auto & flag : flags) {
            if (flag == argv[arg])
                flag.value = !flag.value;
            else
                matched--;
        }
        if (!matched) {
            os << "Incorrect arguement '" << argv[arg] << "'\n";
            printArgsHelp(os);
            return false;
        }
        if (inspect) {
            arg++;
            if (arg == argc) {
                os << "Inspect requires pointer arguement.\n";
                printArgsHelp(os);
                return false;
            }
            std::stringstream ss(argv[arg]);
            ss.unsetf(std::ios::basefield);
            std::size_t ptr_val = 0;
            ss >> ptr_val;
            inspect_set.insert((void *) ptr_val);
            inspect = false;
        }
    }
    return true;
} // parseArgs

void printArgsHelp(std::ostream & os)
{
    os << "Usage:\n\t" << exe;
    for (auto param : flags)
        os << " " << param.sflag;
    os << "\n\nOptions:\n";
    os << std::left;
    for (auto param : flags) {
        os << "\t" << std::setw(4) << param.sflag;
        os << std::setw(10) << param.lflag;
        os << param.description;
        os << std::endl;
    }
    os << std::right;
    os << std::endl;
}

bool getHelp()
{
    return help;
}

bool getAll()
{
    return all;
}

bool getPtr()
{
    return ptr;
}

bool getRefresh()
{
    return refresh;
}

std::set<void *> getInspectPointers()
{
    return inspect_set;
}

bool getTrack()
{
    #ifdef ARAX_DATA_TRACK
    return track;

    #else
    return false;

    #endif
}
