#ifndef UTILS_ARAX_ASSERT_HEADER
#define UTILS_ARAX_ASSERT_HEADER
#include "compat.h"

ARAX_CPP void _arax_assert(int fail, const char *expr, const char *file, int line);

#define arax_assert(EXPR) \
    _arax_assert(!(EXPR),#EXPR, __FILE__, __LINE__)

ARAX_CPP void _arax_assert_obj(void *obj, int type);

#define arax_assert_obj(OBJ, TYPE) \
    ARAX_CPP void _arax_assert_obj(void *obj, int type);

#endif // ifndef UTILS_ARAX_ASSERT_HEADER
