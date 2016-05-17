#ifndef UTILS_QUEUE_HEADER
#define UTILS_QUEUE_HEADER
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct queue;

typedef struct queue utils_queue_s;

/**
 * Initialize a queue at the memory pointed by buff.
 *
 * @param buff Allocated buffer.
 * @param bytes Size of provided buffer to be used.
 * @return queue instance.NULL on failure.
 */
utils_queue_s* utils_queue_init(void *buff, size_t bytes);

/**
 * Calculate byte allocation required for an queue with specified slots.
 *
 * @param slots Number of slots in the queue.
 * @return Size of required buffer size able to fit the slots.
 */
size_t utils_queue_calc_bytes(int slots);

/**
 * Return number of unused slots in the queue.
 *
 * @param q Valid queue instance pointer.
 * @return Number of free slots in queue.
 */
int utils_queue_free_slots(utils_queue_s *q);

/**
 * Return number of used slots in the queue.
 *
 * @param q Valid queue instance pointer.
 * @return Number of used slots in queue.
 */
int utils_queue_used_slots(utils_queue_s *q);

/**
 * Add data to an queue
 *
 * @param q Valid queue instance pointer.
 * @param data Non NULL pointer to data.
 * @return Equal to data, NULL on failure.
 */
void* utils_queue_push(utils_queue_s *q, void *data);

/**
 * Pop data from queue.
 *
 * @param q Valid queue instance pointer.
 * @return Data pointer, NULL on failure.
 */
void* utils_queue_pop(utils_queue_s *q);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef UTILS_QUEUE_HEADER */
