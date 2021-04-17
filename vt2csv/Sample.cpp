#include "Sample.h"

Sample :: Sample(std::vector<const size_t *> & cvalues)
{
    values.reserve(values.size());
    start = get_now();
    for (auto & v : cvalues)
        values.push_back(*v);
    stop = get_now();
}

bool operator == (const Sample & a, const Sample & b)
{
    auto a_itr = a.values.begin();
    auto a_end = a.values.end();

    auto b_itr = b.values.begin();

    for (; a_itr != a_end; a_itr++, b_itr++) {
        if (*a_itr != *b_itr)
            return false;
    }
    return true;
}
