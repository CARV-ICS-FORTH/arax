#ifndef VINE_ACCEL_HEADER
#define VINE_ACCEL_HEADER
#include <vine_talk.h>
#include "utils/list.h"
#include "core/vine_object.h"

typedef struct vine_accel_s vine_accel_s;

#include "core/vine_vaccel.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct vine_accel_s {
	vine_object_s      obj;
	utils_spinlock     lock;
	utils_list_s       vaccels;
	vine_accel_type_e  type;
	vine_accel_loc_s   location;
	vine_accel_stats_s stats;
	vine_accel_state_e state;
	size_t             revision;
	/* To add more as needed */
};

/**
 * Initialize a vine_accel descriptor in the provided \c mem with the provided
 * arguements.
 * @return An initialized vine_accel instance on success, or NULL on failure.
 */
vine_accel_s* vine_accel_init(vine_object_repo_s *repo, void *mem, const char *name,
                              vine_accel_type_e type);

size_t vine_accel_calc_size(const char *name);

const char* vine_accel_get_name(vine_accel_s *accel);

vine_accel_state_e vine_accel_get_stat(vine_accel_s *accel,vine_accel_stats_s * stat);

/*
 * Increase 'revision' of accelerator.
 */
void vine_accel_inc_revision(vine_accel_s * accel);

/*
 * Get 'revision' of accelerator.
 */
size_t vine_accel_get_revision(vine_accel_s * accel);

/**
 * Add (register) a virtual accell \c vaccel to physical accelerator \c accel.
 *
 * @param accel A physsical accelerator
 * @param vaccel A virtual accelerator to be linked with \c accel
 */
void vine_accel_add_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel);

/**
 * Delete (unregister) a virtual accell \c vaccel from physical accelerator \c accel.
 *
 * @param accel A physsical accelerator
 * @param vaccel A virtual accelerator to be unlinked from \c accel
 */
void vine_accel_del_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel);
#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_ACCEL_HEADER */
