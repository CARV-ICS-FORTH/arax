#ifndef VT2CSV_TIMESTAMP
#define VT2CSV_TIMESTAMP

typedef long int Timestamp;

// Get current timestamp, relative to epoch.
Timestamp get_now();

// Reset epoch to current time.
void reset_epoch();
#endif
