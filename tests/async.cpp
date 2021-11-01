#include "async.h"
#include "testing.h"

void* completion_complete_lazy(void *data)
{
    async_completion_s *completion = (async_completion_s *) data;

    safe_usleep(100000);
    async_completion_complete(completion);
    return 0;
}

void* semaphore_inc_lazy(void *data)
{
    async_semaphore_s *sem = (async_semaphore_s *) data;

    safe_usleep(100000);
    async_semaphore_inc(sem);
    return 0;
}

void* cond_thread(void *data)
{
    async_condition_s *cond = (async_condition_s *) data;

    async_condition_lock(cond);
    async_condition_notify(cond);
    async_condition_unlock(cond);
    async_condition_lock(cond);
    async_condition_wait(cond);
    async_condition_unlock(cond);
    return 0;
}

TEST_CASE("async test")
{
    async_meta_s meta;

    test_common_setup();
    // This will not work for ivshmem
    async_meta_init_once(&meta, 0);

    SECTION("serial_completion")
    {
        pthread_t *thread;
        async_completion_s completion;

        async_completion_init(&meta, &completion);
        REQUIRE(!async_completion_check(&completion));
        REQUIRE(!async_completion_check(&completion));
        async_completion_complete(&completion);
        REQUIRE(async_completion_check(&completion));
        REQUIRE(async_completion_check(&completion));
        async_completion_wait(&completion);
        REQUIRE(!async_completion_check(&completion));
        async_completion_init(&meta, &completion);
        thread = spawn_thread(completion_complete_lazy, &completion);
        async_completion_wait(&completion);
        wait_thread(thread);
    }

    SECTION("serial_semaphore")
    {
        pthread_t *thread;
        async_semaphore_s sem;

        async_semaphore_init(&meta, &sem);
        REQUIRE(async_semaphore_value(&sem) == 0);
        async_semaphore_inc(&sem);
        REQUIRE(async_semaphore_value(&sem) == 1);
        async_semaphore_dec(&sem);
        REQUIRE(async_semaphore_value(&sem) == 0);
        thread = spawn_thread(semaphore_inc_lazy, &sem);
        async_semaphore_dec(&sem);
        wait_thread(thread);
    }

    SECTION("serial_condition")
    {
        pthread_t *thread;
        async_condition_s cond;

        async_condition_init(&meta, &cond);
        async_condition_lock(&cond);
        async_condition_unlock(&cond);
        thread = spawn_thread(cond_thread, &cond);
        async_condition_lock(&cond);
        async_condition_wait(&cond);
        async_condition_unlock(&cond);
        async_condition_lock(&cond);
        async_condition_notify(&cond);
        async_condition_unlock(&cond);
        wait_thread(thread);
    }

    test_common_teardown();
    async_meta_exit(&meta);
}
