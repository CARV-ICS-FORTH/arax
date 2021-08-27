#include "testing.h"
#include "vine_pipe.h"      // init test
#include "core/vine_data.h" // free date
#include "core/vine_accel.h"// inc dec size
#include <pthread.h>


#define GPU_SIZE  2097152
#define DATA_SIZE 997152
#define BIG_SIZE  1597152

async_meta_s meta;

void setup()
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x10000000);

    write(fd, config, strlen(config) );

    close(fd);

    // This will not work for ivshmem
    async_meta_init_once(&meta, 0);
}

void teardown()
{
    test_common_teardown();
    async_meta_exit(&meta);
}

void* size_inc(void *accel)
{
    vine_accel_size_inc(accel, DATA_SIZE);
    return 0;
}

void* size_big_inc(void *accel)
{
    vine_accel_size_inc(accel, BIG_SIZE);
    return 0;
}

void* size_big_dec(void *accel)
{
    vine_accel_size_dec(accel, BIG_SIZE);
    return 0;
}

void* size_dec(void *accel)
{
    vine_accel_size_dec(accel, DATA_SIZE);
    return 0;
}

vine_proc_s* create_proc(vine_pipe_s *vpipe, const char *name)
{
    vine_proc_s *proc;

    ck_assert(!!vpipe);
    proc = (vine_proc_s *) vine_proc_register(name);
    return proc;
}

START_TEST(test_gpu_size)
{
    vine_accel_s *accel;
    // init vine_talk
    vine_pipe_s *mypipe = vine_talk_init();

    ck_assert(!!mypipe);

    // init accel
    accel = vine_accel_init(mypipe, "FakeAccel", 1, GPU_SIZE, GPU_SIZE * 2);
    ck_assert(accel != 0);

    // releaseAccelerator
    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);
    vine_accel_release((vine_accel **) &accel);
    ck_assert_ptr_eq(accel, 0);

    // exit vine_talk
    vine_talk_exit();
}
END_TEST START_TEST(test_thread_inc_dec_size_simple)
{
    // staff to use
    pthread_t *thread;
    size_t size_before;
    vine_accel_s *accel, *myaccel;
    vine_accel_type_e accelType = GPU; // GPU : 1

    // init vine_talk
    vine_pipe_s *mypipe = vine_talk_init();

    ck_assert(!!mypipe);

    // create proc
    vine_proc_s *process_id = create_proc(mypipe, "issue_proc");

    ck_assert(!!process_id);

    // initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE, GPU_SIZE * 2);
    ck_assert(accel != 0);

    // acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
    ck_assert(!!myaccel);

    // set phys
    vine_accel_set_physical(myaccel, accel);

    // test inc
    size_before = vine_accel_get_available_size(accel);
    thread      = spawn_thread(size_inc, accel);
    wait_thread(thread);
    // check
    ck_assert_int_eq(vine_accel_get_available_size(accel), size_before + DATA_SIZE);

    // test dec
    size_before = vine_accel_get_available_size(accel);
    thread      = spawn_thread(size_dec, accel);
    wait_thread(thread);
    // check
    ck_assert_int_eq(vine_accel_get_available_size(accel), size_before - DATA_SIZE);


    // exit vine_talk
    vine_talk_exit();
} /* START_TEST */

END_TEST START_TEST(test_thread_wait)
{
    // staff to use
    pthread_t *thread1, *thread2, *thread3, *thread4;
    size_t size_before = 0;
    vine_accel_s *accel, *myaccel;
    vine_accel_type_e accelType = GPU; // GPU : 1

    // init vine_talk
    vine_pipe_s *mypipe         = vine_talk_init();

    ck_assert(!!mypipe);

    // create proc
    vine_proc_s *process_id = create_proc(mypipe, "issue_proc");

    ck_assert(!!process_id);

    // initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE, GPU_SIZE * 2);
    ck_assert(accel != 0);

    // acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
    ck_assert(!!myaccel);

    // set phys
    vine_accel_set_physical(myaccel, accel);

    // first dec
    size_before = vine_accel_get_available_size(accel);
    thread1     = spawn_thread(size_big_dec, accel);
    wait_thread(thread1);
    ck_assert_int_eq(vine_accel_get_available_size(accel), size_before - BIG_SIZE);

    // wait here
    size_before = vine_accel_get_available_size(accel);
    thread1     = spawn_thread(size_big_dec, accel);
    thread3     = spawn_thread(size_big_dec, accel);
    safe_usleep(1000);
    ck_assert_int_eq(vine_accel_get_available_size(accel), size_before);

    thread2 = spawn_thread(size_big_inc, accel);
    safe_usleep(1000);
    ck_assert_int_eq(vine_accel_get_available_size(accel), size_before);

    thread4 = spawn_thread(size_big_inc, accel);
    safe_usleep(1000);
    ck_assert_int_eq(vine_accel_get_available_size(accel), size_before);

    wait_thread(thread4);
    wait_thread(thread3);
    wait_thread(thread2);
    wait_thread(thread1);
    ck_assert_int_eq(vine_accel_get_available_size(accel), GPU_SIZE - BIG_SIZE);

    // exit vine_talk
    vine_talk_exit();
} /* START_TEST */

END_TEST

/*
 * START_TEST(test_assert_false)
 * {
 *  vine_assert(0);
 * }
 * END_TEST
 */

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("Accel Throttle");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
    // add tests here
    tcase_add_test(tc_single, test_gpu_size);
    tcase_add_test(tc_single, test_thread_inc_dec_size_simple);
    tcase_add_test(tc_single, test_thread_wait);
    //    tcase_add_test(tc_single, test_single_phys_task_issue_without_wait);
    // tcase_add_test_raise_signal(tc_single, test_assert_false,6);

    suite_add_tcase(s, tc_single);
    return s;
}

int main(int argc, char *argv[])
{
    Suite *s;
    SRunner *sr;
    int failed;

    s  = suite_init();
    sr = srunner_create(s);
    srunner_set_fork_status(sr, CK_NOFORK);// To debug CK_NOFORK
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
