#ifndef UTILS_VINE_ASSERT_HEADER
#define UTILS_VINE_ASSERT_HEADER
#include "compat.h"

VINE_CPP void _vine_assert(int fail, const char *expr, const char *file, int line);

#define vine_assert(EXPR) \
    _vine_assert(!(EXPR),#EXPR, __FILE__, __LINE__)

VINE_CPP void _vine_assert_obj(void *obj, int type);

#define vine_assert_obj(OBJ, TYPE) \
    VINE_CPP void _vine_assert_obj(void *obj, int type);

#endif // ifndef UTILS_VINE_ASSERT_HEADER
