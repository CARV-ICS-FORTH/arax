#ifndef QUEUE_HEADER
#define QUEUE_HEADER

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct queue;

typedef struct queue queue_s;

/**
 * Initialize a queue at the memory pointed by buff.
 *
 * @param buff Allocated buffer.
 * @param bytes Size of provided buffer to be used.
 * @return queue instance.NULL on failure.
 */
queue_s* queue_init(void *buff, int bytes);

/**
 * Calculate byte allocation required for an queue with specified slots.
 *
 * @param slots Number of slots in the queue.
 * @return Size of required buffer size able to fit the slots.
 */
int queue_calc_bytes(int slots);

/**
 * Return number of unused slots in the queue.
 *
 * @param q Valid queue instance pointer.
 * @return Number of free slots in queue.
 */
int queue_free_slots(queue_s *q);

/**
 * Return number of used slots in the queue.
 *
 * @param q Valid queue instance pointer.
 * @return Number of used slots in queue.
 */
int queue_used_slots(queue_s *q);

/**
 * Add data to an queue
 *
 * @param q Valid queue instance pointer.
 * @param data Non NULL pointer to data.
 * @return Equal to data, NULL on failure.
 */
void* queue_push(queue_s *q, void *data);

/**
 * Pop data from queue.
 *
 * @param q Valid queue instance pointer.
 * @return Data pointer, NULL on failure.
 */
void* queue_pop(queue_s *q);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef QUEUE_HEADER */
