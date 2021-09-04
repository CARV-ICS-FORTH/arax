#ifndef VT2CSV_PHASE
#define VT2CSV_PHASE
#include <string>
#include "Timestamp.h"

class Phase
{
public:
    Phase(std::string title);
    ~Phase();
private:
    Timestamp start;
};
#endif // ifndef VT2CSV_PHASE
