#ifndef VTOP_VALUE_TABLE_HEADER_FILE
#define VTOP_VALUE_TABLE_HEADER_FILE
#include "ValueBar.h"
#include <vector>
#include "VtopWindow.h"
#include "Utils.h"

template <class T, class B>
class ValueTable : public NCursesWindow, public VtopWindow
{
public:
    typedef std::string Formater(const B & val);

    ValueTable(NCursesWindow & parent, int y, bool dense = true)
        : NCursesWindow(parent, 3, parent.width(), y, 0)
        , dense(dense)
        , name_size(3)
        , value_size(1)
    { }

    void addRow(const std::string name, Formater *fmt, const T &value, T max = (T) 1, bool complement = false)
    {
        name_size = std::max(name_size, name.length());
        Resize(true);
        rows.push_back(new B(*this, name, name_size, fmt, value_size, value, max, complement));
    }

    void Draw()
    {
        for (auto & row : rows)
            row->Draw();
        smooth_border(*this);
        if (!dense) {
            for (size_t y = 1; y < rows.size(); y++) {
                hline(2 * y, 1, width() - 2);
                printw(2 * y, 3 + name_size, "\u256B");
                printw(2 * y, width() - (1 + value_size), "\u256B");
            }
        }
        printw(0, 3 + name_size, "\u2565");
        printw(maxy(), 3 + name_size, "\u2568");

        printw(0, width() - (1 + value_size), "\u2565");
        printw(maxy(), width() - (1 + value_size), "\u2568");
    }

    virtual void Resize()
    {
        Resize(false);
    }

    void Resize(bool enlarge)
    {
        size_t new_h = std::max((size_t) 3, 2 + (rows.size() + enlarge) * ( 1 + !dense ) - !dense);

        if (dense)
            wresize(new_h, name_size + value_size + 4);
        else
            wresize(new_h, par->width());

        for (auto & row : rows)
            row->Resize();
        Draw();
    }

private:
    bool dense;
    size_t name_size;
    size_t value_size;
    std::vector<B *> rows;
};
#endif // ifndef VTOP_VALUE_TABLE_HEADER_FILE
