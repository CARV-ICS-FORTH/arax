#include "testing.h"
#include "core/vine_throttle.h"
#include <pthread.h>

void* size_inc(void *thr)
{
    vine_throttle_size_inc((vine_throttle_s *) thr, 10);
    return 0;
}

void* size_dec(void *thr)
{
    vine_throttle_size_dec((vine_throttle_s *) thr, 10);
    return 0;
}

TEST_CASE("vine_throttle test")
{
    async_meta_s meta;

    // This will not work for ivshmem
    async_meta_init_once(&meta, 0);

    // initialize
    vine_throttle_s _obj;
    vine_throttle_s *temp = &_obj;

    REQUIRE(!!temp);

    SECTION("Initialize")
    {
        vine_throttle_init(&meta, temp, 10, 100);
        ///Check
        REQUIRE(temp->available == 10);
        REQUIRE(temp->capacity == 100);
        REQUIRE(vine_throttle_get_available_size(temp) == 10);
        REQUIRE(vine_throttle_get_total_size(temp) == 100);
    }


    for (int size = 0; size < 99; size++) {
        DYNAMIC_SECTION("test_inc_dec Size: " << size)
        {
            vine_throttle_init(&meta, temp, 1000, 10000);
            ///Check init
            REQUIRE(temp->available == 1000);
            REQUIRE(temp->capacity == 10000);
            REQUIRE(vine_throttle_get_available_size(temp) == 1000);
            REQUIRE(vine_throttle_get_total_size(temp) == 10000);
            // check dec
            vine_throttle_size_dec(temp, size);
            REQUIRE(vine_throttle_get_available_size(temp) == 1000 - size);

            vine_throttle_size_inc(temp, size);
            REQUIRE(vine_throttle_get_available_size(temp) == 1000);
        }
    }

    SECTION("test_wait")
    {
        // initialize
        pthread_t *thread1, *thread2, *thread3, *thread4;

        vine_throttle_init(&meta, temp, 15, 1000);
        ///Check
        REQUIRE(vine_throttle_get_available_size(temp) == 15);
        REQUIRE(vine_throttle_get_total_size(temp) == 1000);

        thread1 = spawn_thread(size_dec, temp);
        wait_thread(thread1);
        REQUIRE(vine_throttle_get_available_size(temp) == 5);

        thread1 = spawn_thread(size_dec, temp);
        thread2 = spawn_thread(size_dec, temp);
        safe_usleep(1000);
        REQUIRE(vine_throttle_get_available_size(temp) == 5);

        thread3 = spawn_thread(size_inc, temp);

        thread4 = spawn_thread(size_inc, temp);

        wait_thread(thread4);
        wait_thread(thread3);
        wait_thread(thread2);
        wait_thread(thread1);

        REQUIRE(temp->available == 5);
    }

    /**
     * These tests can not be potred to Catch2 as it has to abort(signal 6).
     * Catch2 at the time of writing does not support signals.
     *
     * SECTION("test_assert_init_1")
     * {
     *  vine_throttle_init(&meta, 0, 10, 100);
     * }
     *
     * SECTION("test_assert_init_2")
     * {
     *  vine_throttle_init(&meta, temp, 0, 100);
     * }
     *
     * SECTION("test_assert_init_3")
     * {
     *  vine_throttle_init(&meta, temp, 10, 0);
     * }
     *
     * SECTION("test_assert_init_4")
     * {
     *  vine_throttle_init(&meta, temp, 100, 10);
     * }
     *
     * SECTION("test_assert_init_5")
     * {
     *  vine_throttle_init(0, temp, 100, 1000);
     * }
     *
     * SECTION("test_assert_get_1")
     * {
     *  vine_throttle_get_available_size(0);
     * }
     *
     * SECTION("test_assert_get_2")
     * {
     *  vine_throttle_get_total_size(0);
     * }
     *
     * SECTION("test_assert_dec_1")
     * {
     *  vine_throttle_size_dec(0, 100);
     * }
     *
     * SECTION("test_assert_inc_1")
     * {
     *  vine_throttle_size_inc(0, 200);
     * }
     */
}
