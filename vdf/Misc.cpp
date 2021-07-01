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

std::string minPtr(void *ptr, int digits)
{
    std::ostringstream oss;

    oss << ptr;
    return oss.str().substr(digits);
}

const char* normalize(const char *label, std::size_t size)
{
    static char buff[1024];

    snprintf(buff, sizeof(buff), "<th>%s</th><td>%s</td>", label, autoBytes(size).c_str());
    return buff;
}

int calcDigits(void *ptr, std::size_t size)
{
    std::ostringstream iss0, iss1;

    iss0 << ptr;
    iss1 << (void *) (((uint8_t *) ptr) + size);

    std::string a = iss0.str(), b = iss1.str();
    int l;

    for (l = 0; l < std::min(a.size(), b.size()); l++) {
        if (a[l] != b[l])
            break;
    }

    return l;
}
