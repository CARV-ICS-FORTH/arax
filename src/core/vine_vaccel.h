#ifndef VINE_VACCEL_HEADER
#define VINE_VACCEL_HEADER
#include "vine_talk_types.h"
#include "utils/queue.h"
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
	uint64_t          cid;
	uint64_t          priority;
	vine_accel_s      *phys;
	void              *meta;	// Metadata pointer available to controller.
	utils_queue_s     queue;
};

/**
 * Initialize a vine_vaccel_s in \c mem.
 *
 * \param repo A valid vine_object_repo_s instance
 * \param name Name of the virtual accelerator
 * \param type Type of the virtual accelerator
 * \param accel A physical accelerator
 */
vine_vaccel_s* vine_vaccel_init(vine_object_repo_s *repo, const char *name,
								vine_accel_type_e  type,vine_accel_s *accel);

/**
 * Set the client id for this virtual accelerator.
 */
uint64_t vine_vaccel_set_cid(vine_vaccel_s *vaccel,uint64_t cid);

/**
 * Get the client id for this virtual accelerator.
 */
uint64_t vine_vaccel_get_cid(vine_vaccel_s *vaccel);

/**
 * Set the priority (latency or throughput critical) for this virtual accelerator.
 */
uint64_t vine_vaccel_set_job_priority(vine_vaccel_s *vaccel,uint64_t priority);

/**
 * Get the priority (latency or throughput critical) for this virtual accelerator.
 */
uint64_t vine_vaccel_get_job_priority(vine_vaccel_s *vaccel);

/**
 * Get the meta for this virtual accelerator.
 */
void vine_vaccel_set_meta(vine_vaccel_s *vaccel,void * meta);

/**
 * Set the meta for this virtual accelerator.
 */
void * vine_vaccel_get_meta(vine_vaccel_s *vaccel);

/**
 * Get the queue of \c vaccel.
 *
 * \param vaccel A virtual accelerator
 * \return The queue of \c vaccel,NULL on failure
 */
utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel);

/**
 * Requrn size of \c vaccel.
 *
 * \param vaccel A virtual accelerator
 * \return The size of the queue of \c vaccel.
 */
unsigned int vine_vaccel_queue_size(vine_vaccel_s *vaccel);

vine_accel_state_e vine_vaccel_get_stat(vine_vaccel_s *accel,vine_accel_stats_s * stat);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_VACCEL_HEADER */
