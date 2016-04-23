/**
 * @brief   A single-producer/single-consumer queue implementation based
 *          on a circular buffer.
 *
 * @details This implementation uses two local states, one for the
 *          producer and another for the consumer.  This way we aim to
 *          reduce the cache-line transfers by reusing the local state
 *          as long as the consumer is aware that there is more work on
 *          the queue and the producer is aware that there is more free
 *          space in the queue.  Each state resides in a different cache
 *          line.  The state comprises of a \c head and a \c tail index
 *          in the circular buffer.  \c head is the location where the
 *          consumer finds elements to consume, while \c tail is the
 *          location where the producer adds new elements.  \c head is
 *          only written by the consumer and \c tail is only written by
 *          the producer.  The consumer keeps a cached value of the \c
 *          tail and the producer keeps a caches value of the \c head to
 *          reduce data transfers, due to the cache coherence protocol.
 *          The cached values get updated when the producer fails to
 *          find free space in the circular buffer or when the consumer
 *          fails to find elements to consume.
 *
 * @author  Foivos Zakkak <zakkak@ics.forth.gr>
 */

#include "queue.h"

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#define COMPILER_BARRIER() asm volatile ("" : : : "memory")

#ifdef __GNUC__
#define UNLIKELY(cond) __builtin_expect(cond, 0)
#define LIKELY(cond)   __builtin_expect(cond, 1)
#else /* ifdef __GNUC__ */
#define UNLIKELY(cond) (cond)
#define LIKELY(cond)   (cond)
#endif /* ifdef __GNUC__ */

/**
 * Internal structure of queue.
 */
struct _state {
	volatile unsigned int tail; /**< Push here */
	volatile unsigned int head; /**< Pop here */
} __attribute__( ( aligned(64) ) );
struct queue {
	struct _state producer; /**< Producer's local state */
	struct _state consumer; /**< Consumer's local state */
	unsigned int  capacity; /**< The capacity of this queue */
	void          *entries[]; /**< Pointers to data. */
} __attribute__( ( aligned(64) ) );
utils_queue_s* utils_queue_init(void *buff, int bytes)
{
	struct queue *ring = buff;

	if ( bytes < sizeof(struct queue)+sizeof(void*) )
		return 0; /* Buffer too small */

	memset( buff, 0, sizeof(struct queue) );
	ring->capacity = ( bytes-sizeof(struct queue) )/sizeof(void*);

	return ring;
}

int utils_queue_calc_bytes(int slots)
{
	return sizeof(struct queue) + ( slots*sizeof(void*) );
}

int utils_queue_used_slots(utils_queue_s *q)
{
	register int used_slots;

	assert( q->capacity < (UINT_MAX/2) );

	used_slots = ( (q->producer.tail+q->capacity) - q->consumer.head ) %
	             q->capacity;

	return used_slots;
}

int utils_queue_free_slots(utils_queue_s *q)
{
	register int free_slots;

	assert( q->capacity < (UINT_MAX/2) );

	free_slots = ( (q->consumer.head+q->capacity) - (q->producer.tail+1) ) %
	             q->capacity;

	return free_slots;
}

/**
 * Check if the consumer can pop an element from \c q (based on its
 * local view of the queue)
 *
 * @param[in] q The queue we want to check
 * @returns 0 if there are elements to pop
 * @returns >0 if there are no elements to pop
 */
static inline int cannot_pop(utils_queue_s *q)
{
	return q->consumer.tail == q->consumer.head;
}

void* utils_queue_pop(utils_queue_s *q)
{
	void *ret_val;

	if ( UNLIKELY( cannot_pop(q) ) ) {
		q->consumer.tail = q->producer.tail;
		if ( UNLIKELY( cannot_pop(q) ) )
			return NULL;
	}

	ret_val = q->entries[q->consumer.head];
	assert( !cannot_pop(q) );
	/* Make sure the head is updated after the retrieval of the element */
	COMPILER_BARRIER();
	q->consumer.head = (q->consumer.head+1) % q->capacity;

	return ret_val;
}

/**
 * Check if the producer can push an element to \c q (based on its
 * local view of the queue)
 *
 * @param[in] q The queue we want to check
 * @returns 0 if there are free slots in the circular buffer
 * @returns >0 if the circular buffer is full
 */
static inline int cannot_push(utils_queue_s *q)
{
	return ( (q->producer.tail+1)%q->capacity ) == q->producer.head;
}

void* utils_queue_push(utils_queue_s *q, void *data)
{
	if ( UNLIKELY( cannot_push(q) ) ) {
		q->producer.head = q->consumer.head;
		if ( UNLIKELY( cannot_push(q) ) )
			return NULL;
	}

	q->entries[q->producer.tail] = data;
	assert( !cannot_push(q) );
	/* Make sure the tail is updated after the addition of the new element
	 * */
	COMPILER_BARRIER();
	q->producer.tail = (q->producer.tail+1) % q->capacity;

	return data;
}
