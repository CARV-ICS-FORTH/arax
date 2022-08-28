#include "testing.h"
#include "arax_pipe.h" // init test
#include <pthread.h>

#define SIZE_INC 3000000
void* size_inc(void *pipe)
{
    arax_pipe_size_inc((arax_pipe_s *) pipe, SIZE_INC);
    return 0;
}

#define SIZE_DEC 3000000
void* size_dec(void *pipe)
{
    arax_pipe_size_dec((arax_pipe_s *) pipe, SIZE_DEC);
    return 0;
}

#define SIZE_DEC_BIG 8000000
void* size_dec_big(void *pipe)
{
    arax_pipe_size_dec((arax_pipe_s *) pipe, SIZE_DEC_BIG);
    return 0;
}

TEST_CASE("pipe throttle tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(10000000);

    write(fd, config, strlen(config) );

    close(fd);

    arax_pipe_s *vpipe = arax_first_init();

    REQUIRE(vpipe);

    SECTION("initial state")
    {
        ///Check
        size_t available = arax_pipe_get_available_size(vpipe);

        REQUIRE(available < 10000000);
        REQUIRE(arax_pipe_get_total_size(vpipe) == available);

        for (int size = 1; size < 10000000; size *= 10) {
            DYNAMIC_SECTION("test_inc_dec #" << size)
            {
                arax_pipe_size_dec(vpipe, size);
                REQUIRE(arax_pipe_get_available_size(vpipe) == available - size);

                arax_pipe_size_inc(vpipe, size);
                REQUIRE(arax_pipe_get_available_size(vpipe) == available);
            }
        }

        SECTION("test_wait")
        {
            // initialize
            pthread_t *thread1, *thread2, *thread3, *thread4;

            thread1 = spawn_thread(size_dec_big, vpipe);
            wait_thread(thread1);
            REQUIRE(available - arax_pipe_get_available_size(vpipe) == SIZE_DEC_BIG);

            thread1 = spawn_thread(size_dec, vpipe);
            thread2 = spawn_thread(size_dec, vpipe);
            safe_usleep(1000);
            REQUIRE(available - arax_pipe_get_available_size(vpipe) == SIZE_DEC_BIG);

            thread3 = spawn_thread(size_inc, vpipe);
            safe_usleep(1000);
            REQUIRE(available - arax_pipe_get_available_size(vpipe) == SIZE_DEC_BIG);

            thread4 = spawn_thread(size_inc, vpipe);
            safe_usleep(1000);
            REQUIRE(available - arax_pipe_get_available_size(vpipe) == SIZE_DEC_BIG);

            wait_thread(thread4);
            wait_thread(thread3);
            wait_thread(thread2);
            wait_thread(thread1);

            REQUIRE(available - arax_pipe_get_available_size(vpipe) == SIZE_DEC_BIG);

            arax_pipe_size_inc(vpipe, 8000000);
        }
    }

    /**
     * These tests can not be potred to Catch2 as it has to abort(signal 6).
     * Catch2 at the time of writing does not support signals.
     *	SECTION("test_assert_dec")
     * {
     *  arax_pipe_size_dec(0, SIZE_DEC);
     * }
     *
     * SECTION("test_assert_inc")
     * {
     *  arax_pipe_size_inc(0, SIZE_INC);
     * }
     *
     * SECTION("test_assert_get_1")
     * {
     *  arax_pipe_get_available_size(0);
     * }
     *
     * SECTION("test_assert_get_2")
     * {
     *  arax_pipe_get_total_size(0);
     * }
     */

    arax_final_exit(vpipe);
    test_common_teardown();
}
