#ifndef VTOP_VALUE_BAR_HEADER_FILE
#define VTOP_VALUE_BAR_HEADER_FILE
#include "Graph.h"
#include <string>
#include "Utils.h"

template <class T>
class ValueBar : public NCursesWindow, public VtopWindow
{
public:
    typedef std::string Formater(const ValueBar<T> & val);
    ValueBar(NCursesWindow & parent, const std::string name, size_t &name_w, Formater *fmt, size_t &value_w,
      const T &value, T max, bool complement = false)
        : NCursesWindow(parent, 1, parent.width() - 2, parent.height() - 2, 1, 'r')
        , name(name)
        , name_w(name_w)
        , fmt(fmt)
        , value_w(value_w)
        , value(value)
        , max(max)
        , complement(complement)
    {
        cs.name       = name;
        cs.val        = &value;
        cs.complement = max;
        ClickManager::addClickHandler(&cs);
    }

    void Draw()
    {
        std::string vstr = fmt(*this);

        value_w = std::max(value_w, vstr.size() + 3);

        int bar_w = std::max(0lu, width() - (6 + name_w + value_w));

        printw(0, 1, "%*s \u2551", (int) name_w, name.c_str());
        if (bar_w > 0) {
            T temp = value;
            if (complement)
                temp = max - value;
            hbraile_line(*this, 0, name_w + 4, bar_w, temp, max);
        }

        printw(0, width() - value_w, "\u2551 %s", vstr.c_str());
    }

    virtual void Resize()
    {
        wresize(1, par->width() - 2);
        cs.sx = begx() + width() - value_w;
        cs.ex = begx() + width();
        cs.sy = begy();
        cs.ey = begy() + 1;
    }

    template <const T mult = 1>
    static std::string Memory(const ValueBar<T> & vb)
    {
        std::ostringstream oss;

        T value = vb.value;

        if (vb.complement)
            value = vb.max - value;
        value *= mult;
        oss << siFormat(value) << "b / " << siFormat(vb.max * mult) << "b";
        return oss.str();
    }

    static std::string Value(const ValueBar<T> & vb)
    {
        std::ostringstream oss;

        oss << vb.value;
        return oss.str();
    }

private:
    const std::string name;
    size_t & name_w;
    Formater *fmt;
    size_t & value_w;
    const T &value;
    const bool complement;
    T max;
    ChangeSource cs;
};
#endif // ifndef VTOP_VALUE_BAR_HEADER_FILE
