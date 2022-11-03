#include "Csv.h"

Csv ::Csv(const char *fname)
    : start(std::chrono::system_clock::now()), ofs(fname){ }

Csv &Csv ::print()
{
    //	std::chrono::duration<unsigned long long,std::nano> tstamp =
    // std::chrono::system_clock::now()-start;  ofs << tstamp.count();
    return (*this);
}
