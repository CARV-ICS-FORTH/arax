#include "arax_assert.h"
#include "core/arax_ptr.h"
#include "core/arax_object.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <execinfo.h>
#include "system.h"

#define ARAX_FILE_PREFIX_LEN (strlen(__FILE__) - 23)
// GCOV_EXCL_START
void _arax_assert(int fail, const char *expr, const char *file, int line)
{
    if (fail) {
        fprintf(stderr, "%s <<\n\narax_assert(%s) @ %s:%d\n\n", system_backtrace(1), expr, file + ARAX_FILE_PREFIX_LEN,
          line);
        abort();
    }
}

void _arax_assert_obj(void *obj, int type)
{
    arax_assert(obj && !"Object was null");
    arax_assert(arax_ptr_valid(obj) && !"Object was not valid  ptr");
    arax_assert(arax_ptr_valid(obj) && !"Object was not valid  ptr");
    arax_object_s *vo = obj;

    arax_assert((vo->type == type) && !"Object type was not the expected");
}

// GCOV_EXCL_STOP
