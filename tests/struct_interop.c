#include "testing.h"
#include "struct_interop.h"

void setup()
{ }

void teardown(){ }

START_TEST(test_struct_interop)
{
    size_t c_sizes[STRUCT_INTEROP_SIZES];
    size_t cpp_sizes[STRUCT_INTEROP_SIZES];

    get_c_sizes(c_sizes);
    get_cpp_sizes(cpp_sizes);

    for (int cnt = 0; cnt < STRUCT_INTEROP_SIZES; cnt++) {
        fprintf(stderr, "Size[%d]: %lu %lu\n", cnt, c_sizes[cnt], cpp_sizes[cnt]);
        ck_assert_int_eq(c_sizes[cnt], cpp_sizes[cnt]);
    }
}
END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("struct_interop");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);

    tcase_add_test(tc_single, test_struct_interop);

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
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
