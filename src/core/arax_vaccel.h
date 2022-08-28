#ifndef ARAX_VACCEL_HEADER
#define ARAX_VACCEL_HEADER
#include "core/arax_object.h"
#include "utils/queue.h"

typedef struct arax_vaccel_s arax_vaccel_s;

#include "core/arax_accel.h"
#include "utils/queue.h"
#include "async.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef enum arax_accel_ordering_e
{
    SEQUENTIAL, // < Tasks in VAQ will run one after the other(no overlap)
    PARALLEL    // < Tasks in VAQ can run in parallel(can overlap).
} arax_accel_ordering_e;

/**
 * Virtual Accelerator
 *
 * Creates a dedicated task queue.
 *
 * @NOTE: arax_accel_s are single producer, multiple consumer.
 */
struct arax_vaccel_s
{
    arax_object_s         obj;
    arax_accel_type_e     type;
    arax_accel_ordering_e ordering;
    utils_spinlock        lock;
    utils_list_node_s     vaccels; // Used in pipe->orphan_vacs or phys->vaccels
    uint64_t              cid;
    uint64_t              priority;
    arax_accel_s *        phys;
    void *                meta; // Metadata pointer available to controller.
    utils_queue_s         queue;
};

/**
 * Initialize a arax_vaccel_s in \c mem.
 *
 * @param pipe Valid arax_pipe_s instance.
 * @param name Name of the virtual accelerator
 * @param type Type of the virtual accelerator
 * @param accel A physical accelerator
 */
arax_vaccel_s* arax_vaccel_init(arax_pipe_s *pipe, const char *name,
  arax_accel_type_e type, arax_accel_s *accel);

void arax_vaccel_add_task(arax_vaccel_s *accel, arax_task *task);

/**
 * Set arax_accel_ordering_e mode to \c ordering of provided \c accel.
 *
 * @param vaccel A virtual accelerator
 */
void arax_vaccel_set_ordering(arax_accel_s *accel, arax_accel_ordering_e ordering);

/**
 * Get arax_accel_ordering_e mode of provided \c accel.
 *
 * @param vaccel A virtual accelerator
 */
arax_accel_ordering_e arax_vaccel_get_ordering(arax_accel_s *accel);

/**
 * Set the client id for this virtual accelerator.
 */
uint64_t arax_vaccel_set_cid(arax_vaccel_s *vaccel, uint64_t cid);

/**
 * Get the client id for this virtual accelerator.
 */
uint64_t arax_vaccel_get_cid(arax_vaccel_s *vaccel);

/**
 * Set the priority (latency or throughput critical) for this virtual accelerator.
 */
uint64_t arax_vaccel_set_job_priority(arax_vaccel_s *vaccel, uint64_t priority);

/**
 * Get the priority (latency or throughput critical) for this virtual accelerator.
 */
uint64_t arax_vaccel_get_job_priority(arax_vaccel_s *vaccel);

/**
 * Get the meta for this virtual accelerator.
 */
void arax_vaccel_set_meta(arax_vaccel_s *vaccel, void *meta);

/**
 * Set the meta for this virtual accelerator.
 */
void* arax_vaccel_get_meta(arax_vaccel_s *vaccel);

/**
 * Get the queue of \c vaccel.
 *
 * @param vaccel A virtual accelerator
 * @return The queue of \c vaccel,NULL on failure
 */
utils_queue_s* arax_vaccel_queue(arax_vaccel_s *vaccel);

/**
 * Requrn size of \c vaccel.
 *
 * @param vaccel A virtual accelerator
 * @return The size of the queue of \c vaccel.
 */
unsigned int arax_vaccel_queue_size(arax_vaccel_s *vaccel);

arax_accel_state_e arax_vaccel_get_stat(arax_vaccel_s *accel, arax_accel_stats_s *stat);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_VACCEL_HEADER */
