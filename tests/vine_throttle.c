#include "testing.h"
#include "core/vine_throttle.h"
#include <pthread.h>

async_meta_s meta;

START_TEST(test_init)
{
    // initialize
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(&meta, temp, 10, 100);
    ///Check
    ck_assert_int_eq(temp->available, 10);
    ck_assert_int_eq(temp->capacity, 100);
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 10);
    ck_assert_int_eq(vine_throttle_get_total_size(temp), 100);
    // free
    free(temp);
    return;
}
END_TEST START_TEST(test_inc_dec)
{
    // initialize
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(&meta, temp, 1000, 10000);
    ///Check init
    ck_assert_int_eq(temp->available, 1000);
    ck_assert_int_eq(temp->capacity, 10000);
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 1000);
    ck_assert_int_eq(vine_throttle_get_total_size(temp), 10000);
    // check dec
    vine_throttle_size_dec(temp, _i);
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 1000 - _i);

    vine_throttle_size_inc(temp, _i);
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 1000);

    // free
    free(temp);
}

END_TEST

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

START_TEST(test_wait)
{
    // initialize
    pthread_t *thread1, *thread2, *thread3, *thread4;
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(&meta, temp, 15, 1000);
    ///Check
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 15);
    ck_assert_int_eq(vine_throttle_get_total_size(temp), 1000);

    thread1 = spawn_thread(size_dec, temp);
    wait_thread(thread1);
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 5);

    thread1 = spawn_thread(size_dec, temp);
    thread2 = spawn_thread(size_dec, temp);
    safe_usleep(1000);
    ck_assert_int_eq(vine_throttle_get_available_size(temp), 5);

    thread3 = spawn_thread(size_inc, temp);

    thread4 = spawn_thread(size_inc, temp);

    wait_thread(thread4);
    wait_thread(thread3);
    wait_thread(thread2);
    wait_thread(thread1);

    ck_assert_int_eq(temp->available, 5);

    // free
    free(temp);
    return;
}
END_TEST START_TEST(test_assert_init_1)
{
    vine_throttle_init(&meta, 0, 10, 100);
}

END_TEST START_TEST(test_assert_init_2)
{
    // initialize
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(&meta, temp, 0, 100);
    // free
    free(temp);
}

END_TEST START_TEST(test_assert_init_3)
{
    // initialize
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(&meta, temp, 10, 0);
    // free
    free(temp);
}

END_TEST START_TEST(test_assert_init_4)
{
    // initialize
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(&meta, temp, 100, 10);
    // free
    free(temp);
}

END_TEST START_TEST(test_assert_init_5)
{
    // initialize
    vine_throttle_s *temp = malloc(sizeof(vine_throttle_s));

    ck_assert(!!temp);
    vine_throttle_init(0, temp, 100, 1000);
    // free
    free(temp);
}

END_TEST START_TEST(test_assert_get_1)
{
    vine_throttle_get_available_size(0);
}

END_TEST START_TEST(test_assert_get_2)
{
    vine_throttle_get_total_size(0);
}

END_TEST START_TEST(test_assert_dec_1)
{
    vine_throttle_size_dec(0, 100);
}

END_TEST START_TEST(test_assert_inc_1)
{
    vine_throttle_size_inc(0, 200);
}

END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("Vine Throttle");
    tc_single = tcase_create("Single");
    // add tests here
    tcase_add_test(tc_single, test_init);
    tcase_add_loop_test(tc_single, test_inc_dec, 0, 99);
    tcase_add_test(tc_single, test_wait);
    // now check asserts
    tcase_add_test_raise_signal(tc_single, test_assert_init_1, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_init_2, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_init_3, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_init_4, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_init_5, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_dec_1, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_inc_1, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_get_1, 6);
    tcase_add_test_raise_signal(tc_single, test_assert_get_2, 6);
    // end of tests
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
    srunner_set_fork_status(sr, CK_FORK);// To debug CK_NOFORK
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
