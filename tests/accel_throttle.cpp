#include "testing.h"
#include "arax_pipe.h"      // init test
#include "core/arax_data.h" // free date
#include "core/arax_accel.h"// inc dec size
#include <pthread.h>


#define GPU_SIZE  2097152
#define DATA_SIZE 997152
#define BIG_SIZE  1597152

void* size_inc(void *phys_accel)
{
    arax_accel_size_inc(phys_accel, DATA_SIZE);
    return 0;
}

void* size_big_inc(void *phys_accel)
{
    arax_accel_size_inc(phys_accel, BIG_SIZE);
    return 0;
}

void* size_big_dec(void *phys_accel)
{
    arax_accel_size_dec(phys_accel, BIG_SIZE);
    return 0;
}

void* size_dec(void *phys_accel)
{
    arax_accel_size_dec(phys_accel, DATA_SIZE);
    return 0;
}

TEST_CASE("accel_throttle")
{
    const char *config = test_create_config(0x10000000);

    test_common_setup(config);

    // init arax
    arax_pipe_s *mypipe = arax_first_init();

    REQUIRE(mypipe);

    SECTION("accel_init")
    {
        // init phys_accel
        arax_accel_s *phys_accel = arax_accel_init(mypipe, "FakeAccel", ANY, GPU_SIZE, GPU_SIZE * 2);

        REQUIRE(phys_accel != 0);
        REQUIRE(arax_accel_get_available_size(phys_accel) == GPU_SIZE);
        REQUIRE(arax_accel_get_total_size(phys_accel) == GPU_SIZE * 2);

        // create proc
        arax_proc_s *process_id = create_proc(mypipe, "issue_proc");

        REQUIRE(!!process_id);

        // acquireAccelerator
        arax_accel_s *myaccel = (arax_accel_s *) arax_accel_acquire_type(ANY);

        REQUIRE(!!myaccel);

        // set phys
        arax_accel_set_physical(myaccel, phys_accel);

        SECTION("test_thread_inc_dec_size_simple")
        {
            // staff to use
            pthread_t *thread;
            size_t size_before;


            // test inc
            size_before = arax_accel_get_available_size(phys_accel);
            thread      = spawn_thread(size_inc, phys_accel);
            wait_thread(thread);
            // check
            REQUIRE(arax_accel_get_available_size(phys_accel) == size_before + DATA_SIZE);

            // test dec
            size_before = arax_accel_get_available_size(phys_accel);
            thread      = spawn_thread(size_dec, phys_accel);
            wait_thread(thread);
            // check
            REQUIRE(arax_accel_get_available_size(phys_accel) == size_before - DATA_SIZE);
        }

        SECTION("test_thread_wait")
        {
            // staff to use
            pthread_t *thread1, *thread2, *thread3, *thread4;
            size_t size_before = 0;

            // first dec
            size_before = arax_accel_get_available_size(phys_accel);
            thread1     = spawn_thread(size_big_dec, phys_accel);
            wait_thread(thread1);
            REQUIRE(arax_accel_get_available_size(phys_accel) == size_before - BIG_SIZE);

            // wait here
            size_before = arax_accel_get_available_size(phys_accel);
            thread1     = spawn_thread(size_big_dec, phys_accel);
            thread3     = spawn_thread(size_big_dec, phys_accel);
            safe_usleep(1000);
            REQUIRE(arax_accel_get_available_size(phys_accel) == size_before);

            thread2 = spawn_thread(size_big_inc, phys_accel);
            safe_usleep(1000);
            REQUIRE(arax_accel_get_available_size(phys_accel) == size_before);

            thread4 = spawn_thread(size_big_inc, phys_accel);
            safe_usleep(1000);
            REQUIRE(arax_accel_get_available_size(phys_accel) == size_before);

            wait_thread(thread4);
            wait_thread(thread3);
            wait_thread(thread2);
            wait_thread(thread1);
            REQUIRE(arax_accel_get_available_size(phys_accel) == GPU_SIZE - BIG_SIZE);
        }

        REQUIRE(arax_object_refs((arax_object_s *) myaccel) == 1);
        arax_accel_release((arax_accel **) &myaccel);
        REQUIRE(myaccel == 0);
        // Only the free vaq from phys remains
        REQUIRE(get_object_count(&(mypipe->objs), ARAX_TYPE_VIRT_ACCEL) == 1);

        REQUIRE(arax_object_refs((arax_object_s *) phys_accel) == 1);
        arax_accel_release((arax_accel **) &phys_accel);
        REQUIRE(phys_accel == 0);
        REQUIRE(get_object_count(&(mypipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);

        REQUIRE(arax_object_refs((arax_object_s *) process_id) == 1);
        arax_proc_put(process_id);
    }

    arax_final_exit(mypipe);

    test_common_teardown();
}
