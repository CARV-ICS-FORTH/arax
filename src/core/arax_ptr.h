#ifndef ARAX_PTR_HEADER
#define ARAX_PTR_HEADER
#include "utils/compat.h"

/**
 * Return true if \c ptr is 'inside' the shared segment ranges.
 */
ARAX_CPP int arax_ptr_valid(const void *ptr);
#endif
