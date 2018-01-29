#ifndef VINE_ACCEL_TYPES_HEADER
#define VINE_ACCEL_TYPES_HEADER
#include "vine_talk_types.h"
#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Test \c type, to ensure it is a valid
 * vine_accel_type_e value.
 *
 * \param type Value to be checked.
 * \return 1 if \c type is  a valid vine_accel_type_e value, 0 otherwise.
 */
int vine_accel_valid_type(vine_accel_type_e type);

/**
 * Convert a vine_accel_type_e value to a human readable string.
 * If \c type not a valid vine_accel_type_e value NULL is returned.
 * NOTE: This function should not be used in critical paths!
 *
 * @return A cahracter representation for the given \c type,NULL on error.
 */
const char * vine_accel_type_to_str(vine_accel_type_e type);

/**
 * Convert a string to the matching vine_accel_type_e value.
 * \c type will be compared ignoring capitalization with the string in
 * types_map variable in vine_accel_types.c.
 *
 * NOTE: This function should not be used in critical paths!
 *
 * @return A value from vine_accel_type_e, if no match is found returns
 * VINE_ACCEL_TYPES
 */
vine_accel_type_e vine_accel_type_from_str(const char * type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif
