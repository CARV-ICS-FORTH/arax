#include "ProcessBar.h"
#include "vine_pipe.h"

void OpenGdb :: OnClick()
{
    std::string cmd = "sudo gdb -p " + std::to_string(pid);

    endwin();
    system(cmd.c_str());
    refresh();
}

void KillProcess :: OnClick()
{
    std::string cmd = "sudo kill -9 " + std::to_string(pid);

    endwin();
    system(cmd.c_str());
    vine_pipe_mark_unmap(vine_talk_init(), pid);
    ungetch('r');
}
