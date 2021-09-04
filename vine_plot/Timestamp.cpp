#include "Timestamp.h"
#include <chrono>

// This is not the unix epoch, but rather
// a time just before the metrics recording
Timestamp epoch;

Timestamp get_now()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count() - epoch;
}

void reset_epoch()
{
    epoch = 0;
    epoch = get_now();
}
