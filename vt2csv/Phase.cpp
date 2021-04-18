#include "Phase.h"
#include <iostream>

Phase :: Phase(std::string title)
    : start(get_now())
{
    std::cerr.width(30);
    std::cerr << title << ": ";
}

Phase :: ~Phase()
{
    std::cerr << "Done [ ";
    std::cerr.width(9);
    std::cerr << abs(start - get_now() / 10) / 100.0 << " ms ]\n";
}
