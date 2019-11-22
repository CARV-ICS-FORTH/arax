#include "vine_throttle.h"
#include "utils/vine_assert.h"
#include "stdio.h"

void vine_throttle_init(async_meta_s * meta,vine_throttle_s* thr, size_t a_sz, size_t t_sz){
    //error check
    vine_assert(meta);
    vine_assert(thr);
    vine_assert( a_sz > 0 );
    vine_assert( t_sz > 0 );
    vine_assert( t_sz >= a_sz);

    //init sizes
    thr->available = a_sz;
    thr->capacity     = t_sz;
    //init async
    async_condition_init(meta,&thr->ready);
}

void vine_throttle_size_inc(vine_throttle_s* thr,size_t sz){
    //error check
    vine_assert(thr);
    vine_assert(sz>0);

    //lock critical section
    async_condition_lock(&(thr->ready));

    //inc avaliable size
    thr->available += sz;

	//check bad use of api
    vine_assert(thr->capacity >= thr->available );

    //notify to stop async_condition_wait
    async_condition_notify(&(thr->ready));

    //unlock critical section
    async_condition_unlock(&(thr->ready));
}


void vine_throttle_size_dec(vine_throttle_s* thr,size_t sz){
    //error check
    vine_assert(thr);
    vine_assert(sz>0);

    //lock critical section
    async_condition_lock(&(thr->ready));

    //wait till there is space to dec coutner
 	while( thr->available < sz )
 		async_condition_wait(&(thr->ready));

    //dec avaliable size
    thr->available -= sz;

    //check bad use of api
    vine_assert(thr->capacity >= thr->available );

    //unlock critical section
 	async_condition_unlock(&(thr->ready));
}


size_t vine_throttle_get_avaliable_size(vine_throttle_s* thr){
    //error check
    vine_assert(thr);
    return thr->available;
}


size_t vine_throttle_get_total_size(vine_throttle_s* thr){
    //error check
    vine_assert(thr);
    return thr->capacity;
}
