#include "Misc.h"
#include <iostream>
std::string autoRange(std::size_t value, const char *units[], std::size_t order, std::size_t precission)
{
    int c = 0;
    std::size_t ret = value;
    float viz;
    std::ostringstream oss;

    while (ret > order * order && units[c]) {
        ret /= order;
        c++;
    }

    if (ret < order) {
        viz = ret;
    } else {
        viz = ret * (1.0 / order);
        c++;
    }

    oss << viz << " " << units[c];
    return oss.str();
}

std::string tag_gen(std::string tag, std::string inner_html, std::string attrs)
{
    std::ostringstream oss;

    oss << "<" << tag << ((attrs != "") ? " " : "") << attrs << ">" << inner_html << "</" << tag << ">";

    return oss.str();
}
