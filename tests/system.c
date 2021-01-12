#include "utils/system.h"
#include "testing.h"
void setup()
{
    test_common_setup();
}

void teardown()
{
    test_common_teardown();
}

START_TEST(test_exec_name)
{
    const char *exec_name      = system_exec_name();
    const char *should_be_name = "system_unit";
    const char *binary_name    = exec_name + strlen(exec_name) - strlen(should_be_name);

    ck_assert_str_eq(binary_name, should_be_name);
}
END_TEST START_TEST(test_thread_id)
{
    ck_assert_int_gt(system_thread_id(), 0);
}

END_TEST START_TEST(test_backtrace)
{
    system_backtrace(0);
}

END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("System");
    tc_single = tcase_create("Single");
    tcase_add_checked_fixture(tc_single, setup, teardown);
    tcase_add_test(tc_single, test_exec_name);
    tcase_add_test(tc_single, test_thread_id);
    tcase_add_test(tc_single, test_backtrace);
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
