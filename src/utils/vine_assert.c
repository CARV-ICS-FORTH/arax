#include "vine_assert.h"
#include "core/vine_ptr.h"
#include "core/vine_object.h"
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

void _vine_assert_obj(void *obj, int type)
{
    vine_assert(obj && !"Object was null");
    vine_assert(vine_ptr_valid(obj) && !"Object was not valid vine ptr");
    vine_assert(vine_ptr_valid(obj) && !"Object was not valid vine ptr");
    vine_object_s *vo = obj;

    vine_assert((vo->type == type) && !"Object type was not the expected");
}

// GCOV_EXCL_STOP
