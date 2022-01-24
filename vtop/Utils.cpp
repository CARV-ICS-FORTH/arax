#include "Utils.h"
#include <fstream>
#include <cstring>

void smooth_border(NCursesWindow & wnd)
{
    wnd.border();
    wnd.printw(0, 0, "\u256D");
    wnd.printw(0, wnd.maxx(), "\u256E");
    wnd.printw(wnd.maxy(), 0, "\u2570");
    wnd.printw(wnd.maxy(), wnd.maxx(), "\u256F");
}

std::string getProcessName(int pid)
{
    std::string cmd = "/proc/" + std::to_string(pid) + "/cmdline";
    std::ifstream ifs(cmd.c_str());

    if (!ifs)
        return "<defunct>";

    ifs >> cmd;
    cmd = cmd.substr(0, strlen(cmd.c_str()));
    return cmd;
}
