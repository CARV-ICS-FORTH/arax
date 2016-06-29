#include "utils/config.h"
#include "testing.h"

#define TEST_KEYS 7

const char *vtalk_keys[TEST_KEYS] = {
	"test1", "test2", "test3", "test4", "test5", "test6", "test7"
};
const char *vtalk_vals[TEST_KEYS] = {
	"0", "1", "test3", "4096", "-4096", "0xFFFFFFFF", "0xFFFFFFFFF"
};

void setup()
{
	int cnt;
	int fd;

	test_backup_config();
	fd = test_open_config();
	for (cnt = 0; cnt < TEST_KEYS; cnt++) {
		write( fd, vtalk_keys[cnt], strlen(vtalk_keys[cnt]) );
		write(fd, " ", 1);
		write( fd, vtalk_vals[cnt], strlen(vtalk_vals[cnt]) );
		write(fd, "\n", 1);
	}
	close(fd);
}

void teardown()
{
	test_restore_config();
}

START_TEST(test_config_get_str) {
	char temp[32];

	ck_assert( utils_config_get_str(vtalk_keys[_i], temp, 32) );
	ck_assert_str_eq(temp, vtalk_vals[_i]);
}
END_TEST

START_TEST(test_config_get_str_fail)
{
	char temp[32];
	int  tret[TEST_KEYS] = {
		0, 0, 1, 0, 0, 0
	};

	ck_assert_int_eq(!!utils_config_get_str(vtalk_vals[_i], temp, 32),
	                 tret[_i]);
}

END_TEST

START_TEST(test_config_get_bool)
{
	int temp;
	int tvals[TEST_KEYS] = {
		0, 1, 0, 0, 0, 0
	};
	int tret[TEST_KEYS] = {
		1, 1, 0, 0, 0, 0
	};

	ck_assert_int_eq(utils_config_get_bool(vtalk_keys[_i], &temp,
	                                      !tvals[_i]), tret[_i]);
	if (tret[_i])
		ck_assert_int_eq(temp, tvals[_i]);
	else
		ck_assert_int_eq(temp, !tvals[_i]);
}

END_TEST

START_TEST(test_config_get_int)
{
	int  temp;
	long tvals[TEST_KEYS] = {
		0, 1, 0, 4096, -4096, 0xFFFFFFFF, 0
	};
	int  tret[TEST_KEYS] = {
		1, 1, 0, 1, 1, 0
	};

	ck_assert_int_eq(utils_config_get_int(vtalk_keys[_i], &temp,
	                                     !tvals[_i]), tret[_i]);
	if (tret[_i])
		ck_assert_int_eq(temp, tvals[_i]);
	else
		ck_assert_int_eq(temp, !tvals[_i]);
}

END_TEST

START_TEST(test_config_no_file)
{
	char *conf_file = test_get_config_file();
	int  temp;

	ck_assert( !unlink(conf_file) ); /* Remove test file*/

	ck_assert( !utils_config_get_int("SHOULD_FAIL", &temp, 0) );

	close( test_open_config() ); /* Recreate it for teardown */
}

END_TEST Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Config");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_loop_test(tc_single, test_config_get_str, 0, TEST_KEYS);
	tcase_add_loop_test(tc_single, test_config_get_str_fail, 0, TEST_KEYS);
	tcase_add_loop_test(tc_single, test_config_get_bool, 0, TEST_KEYS);
	tcase_add_loop_test(tc_single, test_config_get_int, 0, TEST_KEYS);
	tcase_add_test(tc_single, test_config_no_file);
	suite_add_tcase(s, tc_single);
	return s;
}

int main(int argc, char *argv[])
{
	Suite   *s;
	SRunner *sr;
	int     failed;

	s  = suite_init();
	sr = srunner_create(s);
	srunner_run_all(sr, CK_NORMAL);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
