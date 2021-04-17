#ifndef VT2CSV_SAMPLE
#define VT2CSV_SAMPLE
#include <vector>
#include <forward_list>
#include "Timestamp.h"

struct Sample
{
    Timestamp           start;
    Timestamp           stop;
    std::vector<size_t> values;
public:
    Sample(std::vector<const size_t *> & cvalues);
};

bool operator == (const Sample & a, const Sample & b);

typedef std::forward_list<Sample> SampleList;
#endif // ifndef VT2CSV_SAMPLE
