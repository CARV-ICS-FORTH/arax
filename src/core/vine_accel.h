#ifndef VINE_ACCEL_HEADER
#define VINE_ACCEL_HEADER
#include <vine_talk.h>
#include "utils/list.h"
#include "core/vine_object.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct {
	vine_object_s      obj;
	vine_accel_type_e  type;
	vine_accel_loc_s   location;
	vine_accel_stats_s stats;
	vine_accel_state_e state;

	/* To add more as needed */
} vine_accel_s;

/**
 * Initialize a vine_accel descriptor in the provided \c mem with the provided
 * arguements.
 * @return An initialized vine_accel instance on success, or NULL on failure.
 */
vine_accel_s* vine_accel_init(vine_object_repo_s *repo, void *mem, char *name,
                              vine_accel_type_e type);

size_t vine_accel_calc_size(char *name);

const char* vine_accel_get_name(vine_accel_s *accel);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_ACCEL_HEADER */
