#include "Graph.h"
#include "Utils.h"
#include <cmath>
#include <map>
#include <vector>

Graph *graph = 0;

Graph :: Graph(NCursesWindow *parent, NCursesWindow *sibling)
    : NCursesWindow(*parent, sibling->height(), parent->width() - sibling->width(), sibling->begy(),
      sibling->maxx() + 1, 'r')
    , sibling(sibling), name("Click Any Metric To Plot"), max(1), val(0), compliment(0)
{
    graph = this;
}

struct utf
{
    uint32_t ret;
    uint8_t  p1;
};
const char* toBraile(int a, int b)
{
    static utf ret = { 0, 0 };
    uint8_t val    = 0;

    a    = (a < 64) ? (1 << a) : (0);
    b    = (b < 64) ? (1 << b) : (0);
    val  = (a & 7);
    val += (a & 8) << 3;

    val    += (b & 7) << 3;
    val    += (b & 8) << 4;
    ret.ret = 0x80A0E2 + ((val & 63) << 16) + ((val >> 6) << 8);

    return (const char *) &ret;
}

void Graph :: Draw()
{
    smooth_border(*this);

    if (val)
        addSample(*val);

    int visible_data = std::min((width() - 3) * 2, (int) samples.size());
    auto itr         = samples.end() - visible_data;
    std::vector<int> norms(visible_data);

    for (int x = 0 ; x < visible_data ; x++, itr++)
        norms[x] = (2.8 * height() * (1 - *itr / max));

    for (int x = 0 ; x < visible_data ; x += 2) {
        int a = norms[x] % 4;
        if (x + 1 < visible_data && norms[x] / 4 == norms[x + 1] / 4) {
            printw(1 + norms[x] / 4.0, 1 + x / 2, toBraile(a, norms[x + 1] % 4));
        } else {
            printw(1 + norms[x] / 4.0, 1 + x / 2, toBraile(a, 100));
            if ( (x + 1) < visible_data)
                printw(1 + norms[x + 1] / 4.0, 1 + x / 2, toBraile(100, norms[x + 1] % 4));
        }
        for (int y = 0 ; y < height() - 2 ; y++)
            if (y != a)
                printw(y + 1, 1 + x / 2, toBraile(100, 100));
    }

    printw(1, 2, " %s ", name.c_str());
}

void Graph :: Resize()
{
    int visible_width = par->width() - sibling->width();

    wresize(sibling->height(), visible_width);
}

void Graph :: Reset()
{
    max = 1;
    samples.clear();
}

void Graph :: setSource(std::string name, const size_t *val, size_t compliment)
{
    if (val != this->val) {
        Reset();
        this->val        = val;
        this->compliment = compliment;
        this->name       = name;
    }
}

void Graph :: addSample(size_t val)
{
    if (val > max)
        max = val;
    samples.push_back(val);
    if (samples.size() > (size_t) (par->width() * 2) )
        samples.pop_front();
}

void ChangeSource :: OnClick()
{
    graph->setSource(name, val, complement);
}
