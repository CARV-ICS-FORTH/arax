#include "utils/vine_plot.h"
#include "testing.h"

void setup()
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);
}

void teardown()
{
    test_common_teardown();
}

START_TEST(test_with_null)
{
    ck_assert(vine_plot_register_metric("test", 0));
}
END_TEST START_TEST(test_with_shm_ptr)
{
    vine_pipe_s *vpipe = vine_talk_init();
    uint64_t *ptr      = arch_alloc_allocate(&(vpipe->allocator), sizeof(uint64_t));

    ck_assert(vine_plot_register_metric("test", ptr));
}

END_TEST START_TEST(test_with_bad_ptr)
{
    vine_plot_register_metric("test", malloc(sizeof(uint64_t)));
}

END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("vine_plot");
    tc_single = tcase_create("Single");
    tcase_add_checked_fixture(tc_single, setup, teardown);
    tcase_add_test(tc_single, test_with_null);
    tcase_add_test(tc_single, test_with_shm_ptr);
    tcase_add_test_raise_signal(tc_single, test_with_bad_ptr, 6);
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

    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
