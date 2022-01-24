#ifndef VTOP_MAIN_VIEW_HEADER_FILE
#define VTOP_MAIN_VIEW_HEADER_FILE
#include "ValueTable.h"
#include "ProcessBar.h"
#include "VtopWindow.h"
#include "Graph.h"

/*
 * Create the following layout
 * ┌─────────────────────────────────┐
 * │              Title              │
 * │             Mem bars            │
 * ├─────────────────┬───────────────┤
 * │   Object Stats  │   Time Graph  │
 * ├─────────────┬───┴──┬────────────┤
 * │  Processes  │ Vaqs │    Tasks   │
 * └─────────────┴──────┴────────────┘
 */

class MainView : public NCursesWindow, public VtopWindow {
public:
    MainView();
    void Setup();
    void Cleanup();
    virtual ~MainView();
    void addMemBar();
    void Show();
    void Draw();
    void Resize();
private:
    bool run;
    int refresh_ms;
    void Update();
    void drawTitle();
    NCursesWindow *title;
    ValueTable<size_t, ValueBar<size_t> > *mem_bars;
    ValueTable<size_t, ValueBar<size_t> > *obj_stats;
    Graph *time_graph;
    ValueTable<size_t, ProcessBar<size_t> > *processes;
};
#endif // ifndef VTOP_MAIN_VIEW_HEADER_FILE
