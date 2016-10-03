#ifndef VINE_VACCEL_HEADER
#define VINE_VACCEL_HEADER
#include <vine_talk.h>
#include "utils/queue.h"
#include "utils/list.h"
#include "core/vine_object.h"

typedef struct vine_vaccel_s vine_vaccel_s;

#include "core/vine_accel.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Virtual Accelerator
 *
 * Creates a dedicated queue mapped to a physical accelerator.
 */
struct vine_vaccel_s {
	vine_object_s     obj;
	vine_accel_type_e type;
	utils_list_node_s vaccels;
	utils_spinlock    lock;
	vine_accel_s      *phys;
};

/**
 * Initialize a vine_vaccel_s in \c mem.
 *
 * \param repo A valid vine_object_repo_s instance
 * \param mem An allocated memory buffer
 * \param mem_size The size of the \c mem buffer in bytes
 * \param name Name of the virtual accelerator
 * \param type Type of the virtual accelerator
 * \param accel A physical accelerator
 */
vine_vaccel_s* vine_vaccel_init(vine_object_repo_s *repo, void *mem,
								size_t mem_size, const char *name,
								vine_accel_type_e  type,vine_accel_s *accel);

/**
 * Get the queue of \c vaccel.
 *
 * \param vaccel A virtual accelerator
 * \return The queue of \c vaccel,NULL on failure
 */
utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel);

vine_accel_state_e vine_vaccel_get_stat(vine_vaccel_s *accel,vine_accel_stats_s * stat);

/**
 * Erase \c accel from the list of virtual accelerators.
 *
 * Will also make \c accel eligible for vine_vaccel_erase().
 *
 * \param repo A valid vine_object_repo_s instance
 * \param accel The virtual accelerator to be erased
 * \return Non zero if successfull
 */
int vine_vaccel_erase(vine_object_repo_s *repo, vine_vaccel_s *accel);

/**
 * Reclaim resources of \c accel.
 *
 * \param accel Virtual accelerator to be reclaimed
 * \return Non zero if \c accel was successfully reclaimed
 */
int vine_vaccel_reclaim(arch_alloc_s *alloc,vine_vaccel_s *accel);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_VACCEL_HEADER */
