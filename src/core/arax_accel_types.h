#ifndef ARAX_ACCEL_TYPES_HEADER
#define ARAX_ACCEL_TYPES_HEADER
#include "arax_types.h"
#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Test \c type, to ensure it is a valid
 * arax_accel_type_e value.
 *
 * @param type Value to be checked.
 * @return 1 if \c type is  a valid arax_accel_type_e value, 0 otherwise.
 */
int arax_accel_valid_type(arax_accel_type_e type);

/**
 * Convert a arax_accel_type_e value to a human readable string.
 * If \c type not a valid arax_accel_type_e value NULL is returned.
 * NOTE: This function should not be used in critical paths!
 *
 * @return A character representation for the given \c type,NULL on error.
 */
const char* arax_accel_type_to_str(arax_accel_type_e type);

/**
 * Convert a string to the matching arax_accel_type_e value.
 * \c type will be compared ignoring capitalization with the string in
 * types_map variable in arax_accel_types.c.
 *
 * NOTE: This function should not be used in critical paths!
 *
 * @return A value from arax_accel_type_e, if no match is found returns
 * ARAX_ACCEL_TYPES
 */
arax_accel_type_e arax_accel_type_from_str(const char *type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef ARAX_ACCEL_TYPES_HEADER
