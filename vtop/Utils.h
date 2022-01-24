#ifndef VTOP_UTILS_HEADER
#define VTOP_UTILS_HEADER
#include <sstream>
#include <string>
#include <cursesw.h>

template <class T>
std::string siFormat(T val)
{
    double fv = val;
    int mag   = 0;

    while (fv > 999) {
        mag++;
        fv /= 1024;
    }
    const char *mags = " kmgtpezy";
    std::ostringstream oss;

    oss << (((T) (fv * 100)) / 100.0) << mags[mag];
    return oss.str();
}

void smooth_border(NCursesWindow & wnd);

template <class T>
void hbraile_line(NCursesWindow & wnd, int y, int x, int width, T val, T max)
{
    const char *bar_map[] = { "\u2840", "\u2844", "\u2846", "\u2847", "\u28C7", "\u28E7", "\u28F7", "\u28FF" };
    T n   = (width * val * 8) / max;
    T mod = n % 8;
    T w   = n / 8;
    T dx  = 0;

    //    std::wstring bar(n,L"\u283F");
    //    wnd.printw(y,x,"%ls",bar.c_str());
    for (; dx < w; dx++)
        wnd.printw(y, dx + x, "\u28FF ");
    if (mod)
        wnd.printw(y, dx + x, bar_map[mod]);
}

std::string getProcessName(int pid);

#endif // ifndef VTOP_UTILS_HEADER
