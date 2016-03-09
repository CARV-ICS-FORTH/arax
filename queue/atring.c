#include "atring.h"
#include <stdint.h>
#include <string.h>

/**
 * Internal structure of atring.
 */
struct _atring {
	volatile uint64_t    bottom __attribute__((aligned(64)));/**< Push here */
	volatile uint64_t    top __attribute__((aligned(64)));	/**< Pop here   */
	uint64_t            slots;           /**< Number of slots in this queue */
	void *entries[];                     /**< Pointers to data. */
} __attribute__((aligned(64)));

atring * atring_init(void * buff,int bytes)
{
	struct _atring * ring = buff;
	if(bytes < slotsof(struct _atring)+slotsof(void*))
		return 0;	/* Buffer too small */
	memset(buff,0,slotsof(struct _atring));
	ring->slots = (bytes-slotsof(struct _atring))/slotsof(void*);
	bytes = 1;
	/* Make it a power of 2 */
	while(bytes <= ring->slots)
		bytes <<= 1;
	bytes >>= 1;
	ring->slots = bytes;
	return buff;
}

int atring_calc_bytes(int slots)
{
	return (slotsof(struct _atring)+(slots*slotsof(void*)));
}

int atring_used_slots(atring * ar)
{
	struct _atring * ring =  ar;
	return (ring->bottom ^ ring->top);
}

int atring_free_slots(atring * ar)
{
	struct _atring * ring =  ar;
	return ring->slots - atring_used_slots(ar);
}

void * atring_pop(atring* ar)
{
	register uint64_t t, b;
	register uint64_t    i;
	void * ret_val;
	struct _atring * ring =  ar;

	/* Only one thief can succeed in the following critical section */
	t = ring->top;
	b = ring->bottom;

	/* If it is empty */
	if (b == t || (uint64_t)(b + 1) == t)
		return 0;

	/* Get the top element */
	i = t & (ring->slots - 1);
	ret_val = ring->entries[i];

	if (__sync_bool_compare_and_swap(&ring->top, t, t + 1)) {

		return ret_val;
	} else {
		//Retry?
		return 0;
	}
}

void * atring_push (atring * ar,void * data)
{
	uint64_t b, t;
	int     i;
	struct _atring * ring =  ar;

	b = ring->bottom;
	t = ring->top;

	/* If there is no more space */
	if ((b ^ t) == ring->slots) {
		return 0;
	}

	i                 = b & (ring->slots - 1);
	ring->entries[i] = data;
	__sync_synchronize();
	ring->bottom     = b + 1;

	return data;
}
