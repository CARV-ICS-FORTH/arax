#ifndef VTOP_GRAPH_HEADER_FILE
#define VTOP_GRAPH_HEADER_FILE
#include "VtopWindow.h"
#include <cursesw.h>
#include <deque>
#include <string>
#include "ClickManager.h"

class Graph : public NCursesWindow, public VtopWindow
{
public:
    Graph(NCursesWindow *parent, NCursesWindow *sibling);
    void Draw();
    void Resize();
    void Reset();
    void setSource(std::string name, const size_t *val, size_t compliment = 0);
private:
    NCursesWindow *sibling;
    std::string name;
    float max;
    const size_t *val;
    size_t compliment;

    std::deque<size_t> samples;

    void addSample(size_t val);
};

struct ChangeSource : public ClickManager::ClickHandler
{
    std::string   name;
    const size_t *val;
    size_t        complement;
    void OnClick();
};

#endif // ifndef VTOP_GRAPH_HEADER_FILE
