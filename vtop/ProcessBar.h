#ifndef VTOP_PROCESS_BAR_HEADER_FILE
#define VTOP_PROCESS_BAR_HEADER_FILE
#include <cursesw.h>
#include <string>
#include "VtopWindow.h"
#include "Utils.h"
#include "ClickManager.h"

struct OpenGdb : public ClickManager :: ClickHandler
{
    OpenGdb(const size_t & pid)
        : pid(pid)
    { }

    void OnClick();
    const size_t & pid;
};

struct KillProcess : public ClickManager :: ClickHandler
{
    KillProcess(const size_t & pid)
        : pid(pid)
    { }

    void OnClick();
    const size_t & pid;
};

template <class T>
class ProcessBar : public NCursesWindow, public VtopWindow
{
public:
    typedef std::string Formater(const ProcessBar<T> & val);
    ProcessBar(NCursesWindow & parent, const std::string name, size_t &name_w, Formater *fmt, size_t &value_w,
      const T &value, T max, bool unused)
        : NCursesWindow(parent, 1, parent.width() - 2, parent.height() - 2, 1, 'r')
        , gdb(value)
        , kill(value)
        , name(name)
        , name_w(name_w)
        , fmt(fmt)
        , value_w(value_w)
        , value(value)
        , max(max)
    {
        ClickManager::addClickHandler(&gdb);
        ClickManager::addClickHandler(&kill);
    }

    void Draw()
    {
        std::string vstr = fmt(*this);

        value_w = std::max(value_w, vstr.size() + 5);

        printw(0, 1, "%*s \u2551", (int) name_w, name.c_str());

        attron(A_BOLD);
        printw(0, 4 + name_w, "\u2613");
        attroff(A_BOLD);

        attron(A_BOLD | A_UNDERLINE);
        printw(0, 4 + width() - value_w, "%s", vstr.c_str());
        attroff(A_BOLD | A_UNDERLINE);
    }

    virtual void Resize()
    {
        wresize(1, par->width() - 2);

        // Update gdb hitbox
        gdb.sx = begx() + 1;
        gdb.ex = begx() + name_w + 1;
        gdb.sy = begy();
        gdb.ey = begy() + 1;

        // Update kill hitbox
        kill.sx = begx() + 4 + width() - value_w;
        kill.ex = begx() + 4 + width() - 5;
        kill.sy = begy();
        kill.ey = begy() + 1;
    }

    template <const T mult = 1>
    static std::string Memory(const ProcessBar<T> & vb)
    {
        std::ostringstream oss;

        oss << siFormat(vb.value * mult) << "b / " << siFormat(vb.max * mult) << "b";
        return oss.str();
    }

    static std::string Value(const ProcessBar<T> & vb)
    {
        std::ostringstream oss;

        oss << vb.value;
        return oss.str();
    }

private:
    OpenGdb gdb;
    KillProcess kill;
    const std::string name;
    size_t & name_w;
    Formater *fmt;
    size_t & value_w;
    const T &value;
    T max;
};
#endif // ifndef VTOP_PROCESS_BAR_HEADER_FILE
