#include "testing.h"
#include "core/arax_throttle.h"
#include <pthread.h>

void* size_inc(void *thr)
{
    arax_throttle_size_inc((arax_throttle_s *) thr, 10);
    return 0;
}

void* size_dec(void *thr)
{
    arax_throttle_size_dec((arax_throttle_s *) thr, 10);
    return 0;
}

TEST_CASE("arax_throttle test")
{
    async_meta_s meta;

    // This will not work for ivshmem
    async_meta_init_once(&meta, 0);

    // initialize
    arax_throttle_s _obj;
    arax_throttle_s *temp = &_obj;

    REQUIRE(!!temp);

    SECTION("Initialize")
    {
        arax_throttle_init(&meta, temp, 10, 100);
        ///Check
        REQUIRE(temp->available == 10);
        REQUIRE(temp->capacity == 100);
        REQUIRE(arax_throttle_get_available_size(temp) == 10);
        REQUIRE(arax_throttle_get_total_size(temp) == 100);
    }


    for (int size = 0; size < 99; size++) {
        DYNAMIC_SECTION("test_inc_dec Size: " << size)
        {
            arax_throttle_init(&meta, temp, 1000, 10000);
            ///Check init
            REQUIRE(temp->available == 1000);
            REQUIRE(temp->capacity == 10000);
            REQUIRE(arax_throttle_get_available_size(temp) == 1000);
            REQUIRE(arax_throttle_get_total_size(temp) == 10000);
            // check dec
            arax_throttle_size_dec(temp, size);
            REQUIRE(arax_throttle_get_available_size(temp) == 1000 - size);

            arax_throttle_size_inc(temp, size);
            REQUIRE(arax_throttle_get_available_size(temp) == 1000);
        }
    }

    SECTION("test_wait")
    {
        // initialize
        pthread_t *thread1, *thread2, *thread3, *thread4;

        arax_throttle_init(&meta, temp, 15, 1000);
        ///Check
        REQUIRE(arax_throttle_get_available_size(temp) == 15);
        REQUIRE(arax_throttle_get_total_size(temp) == 1000);

        thread1 = spawn_thread(size_dec, temp);
        wait_thread(thread1);
        REQUIRE(arax_throttle_get_available_size(temp) == 5);

        thread1 = spawn_thread(size_dec, temp);
        thread2 = spawn_thread(size_dec, temp);
        safe_usleep(1000);
        REQUIRE(arax_throttle_get_available_size(temp) == 5);

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
     *  arax_throttle_init(&meta, 0, 10, 100);
     * }
     *
     * SECTION("test_assert_init_2")
     * {
     *  arax_throttle_init(&meta, temp, 0, 100);
     * }
     *
     * SECTION("test_assert_init_3")
     * {
     *  arax_throttle_init(&meta, temp, 10, 0);
     * }
     *
     * SECTION("test_assert_init_4")
     * {
     *  arax_throttle_init(&meta, temp, 100, 10);
     * }
     *
     * SECTION("test_assert_init_5")
     * {
     *  arax_throttle_init(0, temp, 100, 1000);
     * }
     *
     * SECTION("test_assert_get_1")
     * {
     *  arax_throttle_get_available_size(0);
     * }
     *
     * SECTION("test_assert_get_2")
     * {
     *  arax_throttle_get_total_size(0);
     * }
     *
     * SECTION("test_assert_dec_1")
     * {
     *  arax_throttle_size_dec(0, 100);
     * }
     *
     * SECTION("test_assert_inc_1")
     * {
     *  arax_throttle_size_inc(0, 200);
     * }
     */
}
