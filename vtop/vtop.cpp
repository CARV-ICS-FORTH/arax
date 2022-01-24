#include <locale.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <cstring>
#include <vine_talk.h>
#include <vine_pipe.h>
#include <iostream>
#include "Utils.h"
#include "MainView.h"
#include "ValueTable.h"

void start()
{
    MainView mview;

    mview.Show();
}

void args_to_set(int argc, char *argv[], std::set<std::string> & map)
{
    bool should_ignore = false;

    for (int arg = 0 ; arg < argc ; arg++)
        map.insert(argv[arg]);
}

void ignore_controller()
{
    // Pretend that vtop is the controller
    vine_talk_controller_init_start();
    vine_talk_controller_init_done();
}

void show_help(char *argv[])
{
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << argv[0] << " -i -h" << std::endl;
    std::cout << std::endl << "\t" << "-i --ignore\tIgnore controller initialization" << std::endl;
    std::cout << "\t" << "-h --help  \tShow this help message" << std::endl << std::endl;
}

int main(int argc, char *argv[])
{
    std::set<std::string> flags;

    args_to_set(argc, argv, flags);

    if (flags.count("-i") || flags.count("--ignore"))
        ignore_controller();

    if (flags.count("-h") || flags.count("--help")) {
        show_help(argv);
        return 0;
    }

    setlocale(LC_ALL, "C.UTF-8");

    start();

    return 0;
} // main
