#include "async.h"

void async_meta_init_always(async_meta_s *meta)
{ }

void async_meta_init_once(async_meta_s *meta, arch_alloc_s *alloc)
{ }

void async_completion_init(async_meta_s *meta, async_completion_s *completion)
{
    completion->completed = 0;
}

void async_completion_complete(async_completion_s *completion)
{
    __sync_bool_compare_and_swap(&(completion->completed), 0, 1);
}

void async_completion_wait(async_completion_s *completion)
{
    while (!__sync_bool_compare_and_swap(&(completion->completed), 1, 0));
}

int async_completion_check(async_completion_s *completion)
{
    return completion->completed;
}

void async_semaphore_init(async_meta_s *meta, async_semaphore_s *sem)
{
    sem->value = 0;
}

int async_semaphore_value(async_semaphore_s *sem)
{
    return sem->value;
}

void async_semaphore_inc(async_semaphore_s *sem)
{
    __sync_fetch_and_add(&(sem->value), 1);
}

void async_semaphore_dec(async_semaphore_s *sem)
{
    int value;

    do{
        value = sem->value;
        if (value >= 1) {
            if (__sync_bool_compare_and_swap(&(sem->value), value, value - 1))
                break;
        }
    }while(1);
}

void async_condition_init(async_meta_s *meta, async_condition_s *cond)
{
    async_completion_init(meta, &(cond->mutex));
    async_completion_complete(&(cond->mutex));
    async_semaphore_init(meta, &(cond->semaphore));
}

void async_condition_lock(async_condition_s *cond)
{
    async_completion_wait(&(cond->mutex));
}

void async_condition_wait(async_condition_s *cond)
{
    async_completion_complete(&(cond->mutex));
    async_semaphore_dec(&(cond->semaphore));
    async_completion_wait(&(cond->mutex));
}

void async_condition_notify(async_condition_s *cond)
{
    async_semaphore_inc(&(cond->semaphore));
}

void async_condition_unlock(async_condition_s *cond)
{
    async_completion_complete(&(cond->mutex));
}

void async_meta_exit(async_meta_s *meta)
{ }
