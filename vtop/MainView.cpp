#include "MainView.h"
#include <vine_talk.h>
#include <vine_pipe.h>
#include <stdexcept>

MainView :: MainView()
    : NCursesWindow(stdscr)
    , run(true)
    , refresh_ms(100)
{
    raw();
    keypad(TRUE);
    noecho();
    curs_set(FALSE);
    wtimeout(stdscr, refresh_ms);
    mousemask(BUTTON1_CLICKED, 0);

    Setup();
}

void MainView :: Setup()
{
    endwin();
    vine_pipe_s *vpipe = vine_talk_init();

    if (!vpipe)
        throw std::runtime_error("Vinetalk initialization faild");

    title    = new NCursesWindow(*this, 1, width() - 1, 0, 1);
    mem_bars = new ValueTable<size_t, ValueBar<size_t> >(*this, 1, false);

    mem_bars->addRow("Shm", ValueBar<size_t>::Memory<1>, vpipe->throttle.available, vpipe->throttle.capacity, true);
    utils_bitmap_s *vbmp = arch_alloc_get_bitmap();

    if (vbmp)
        mem_bars->addRow("Bmp", ValueBar<size_t>::Memory<4096>, vbmp->used_bits, vbmp->size_bits);

    for (vine_object_type_e type = (vine_object_type_e) 0;
      type < VINE_TYPE_COUNT;
      type = (vine_object_type_e) ((int) type + 1))
    {
        utils_list_s *olist = vine_object_list_lock(&(vpipe->objs), type);
        if (type == VINE_TYPE_PHYS_ACCEL) {
            utils_list_node_s *itr;
            utils_list_for_each(*olist, itr)
            {
                vine_accel_s *phys = (vine_accel_s *) (itr->owner);

                mem_bars->addRow(phys->obj.name, ValueBar<size_t>::Memory<1>, phys->throttle.available,
                  phys->throttle.capacity, true);
            }
        }
        vine_object_list_unlock(&(vpipe->objs), type);
    }

    obj_stats = new ValueTable<size_t, ValueBar<size_t> >(*this, mem_bars->height() + mem_bars->begy());

    for (vine_object_type_e type = (vine_object_type_e) 0;
      type < VINE_TYPE_COUNT;
      type = (vine_object_type_e) ((int) type + 1))
    {
        utils_list_s *olist = vine_object_list_lock(&(vpipe->objs), type);
        obj_stats->addRow(vine_object_type_to_str(type), ValueBar<size_t>::Value, olist->length);
        vine_object_list_unlock(&(vpipe->objs), type);
    }

    obj_stats->addRow("Clients", ValueBar<size_t>::Value, vpipe->processes);

    time_graph = new Graph(this, obj_stats);

    processes = new ValueTable<size_t, ProcessBar<size_t> >(*this, obj_stats->height() + obj_stats->begy());

    for (int c = 0 ; c < VINE_PROC_MAP_SIZE ; c++) {
        if (vpipe->proc_map[c] != 0)
            processes->addRow(getProcessName(vpipe->proc_map[c]), ProcessBar<size_t>::Value, vpipe->proc_map[c]);
    }
    Resize();
} // MainView::Setup

void MainView ::Cleanup()
{
    endwin();
    vine_talk_exit();
    erase();
    delete title;
    delete mem_bars;
    delete obj_stats;
    delete time_graph;
    delete processes;
    ClickManager::Reset();
}

MainView :: ~MainView()
{
    endwin();
    vine_talk_exit();
}

void MainView :: Show()
{
    erase();
    Resize();
    Draw();
    while (run) {
        Update();
    }
}

void MainView :: Draw()
{
    drawTitle();
    mem_bars->Draw();
    obj_stats->Draw();
    time_graph->Draw();
    processes->Draw();
    refresh();
}

void MainView :: Resize()
{
    wresize(lines(), cols());
    title->wresize(1, width() - 1);
    mem_bars->Resize();
    obj_stats->Resize();
    time_graph->Resize();
    processes->Resize();
}

void MainView :: Update()
{
    switch (getch()) {
        case KEY_RESIZE:
            Resize();
            break;
        case 'q':
        case 'Q':
            run = false;
            break;
        case 'r':
        case 'R':
            endwin();
            Cleanup();
            Setup();
            Resize();
            break;
        case '+':
            refresh_ms = std::min(500, refresh_ms + 10);
            wtimeout(stdscr, refresh_ms);
            break;
        case '-':
            refresh_ms = std::max(10, refresh_ms - 10);
            wtimeout(stdscr, refresh_ms);
            break;
        case KEY_MOUSE: {
            MEVENT event;
            if (getmouse(&event) == OK) {
                ClickManager::Click(event.x, event.y);
                refresh();
            }
            break;
        }
        default: break;
    }

    /*    if(pipe->processes == 1)
     *  {
     *          endwin();
     *          Cleanup();
     *          Setup();
     *          Resize();
     *  }
     */
    erase();
    Draw();
} // MainView::Update

void MainView :: drawTitle()
{
    const std::string title_str = "VTOP";
    const std::string help      = "(Q) Quit (R) Re-attach";
    const size_t help_len       = help.length() + 1;

    title->printw(0, 0, "%s", title_str.c_str());
    title->printw(0, maxx() - help_len, "%s", help.c_str());
}
