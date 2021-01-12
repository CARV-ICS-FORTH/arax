#include "vine_assert.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <execinfo.h>
#include "system.h"

#define VINE_FILE_PREFIX_LEN (strlen(__FILE__) - 23)
// GCOV_EXCL_START
void _vine_assert(int fail, const char *expr, const char *file, int line)
{
    if (fail) {
        fprintf(stderr, "%s <<\n\nvine_assert(%s) @ %s:%d\n\n", system_backtrace(1), expr, file + VINE_FILE_PREFIX_LEN,
          line);
        abort();
    }
}

// GCOV_EXCL_STOP
