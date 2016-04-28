#include "utils/config.h"
#include <check.h>
#include <stdio.h>
#include <pwd.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

extern char* get_home_path(); /* Politicaly incorrect */

#define TEST_KEYS 3
const char * vtalk_keys[TEST_KEYS] =
{"test1","test2","test3"}
;

const char * vtalk_vals[TEST_KEYS] =
{"0","1","test3"}
;

void setup()
{
	char vtpath[1024];
	int fd,cnt;
	snprintf(vtpath,1024,"%s/.vinetalk",get_home_path());
	rename(vtpath,"vinetalk.bak"); /* Keep old file */
	/* Write test file */
	fd = open(vtpath,O_RDWR|O_CREAT,0666);
	ck_assert_int_gt(fd,0);
	for(cnt = 0 ; cnt < TEST_KEYS ; cnt++)
	{
		write(fd,vtalk_keys[cnt],strlen(vtalk_keys[cnt]));
		write(fd," ",1);
		write(fd,vtalk_vals[cnt],strlen(vtalk_vals[cnt]));
		write(fd,"\n",1);
	}
}

void teardown()
{
	char vtpath[1024];
	snprintf(vtpath,1024,"%s/.vinetalk",get_home_path());
	unlink(vtpath);					/* Remove test file*/
	rename("vinetalk.bak",vtpath); /* Revert old file */
}

START_TEST(test_config_get_str)
{
	char temp[32];
	ck_assert(util_config_get_str(vtalk_keys[_i],temp,32));
	ck_assert_str_eq(temp,vtalk_vals[_i]);
}
END_TEST

START_TEST(test_config_get_str_fail)
{
	char temp[32];
	int tret[TEST_KEYS] = {0,0,1};
	ck_assert_int_eq(!!util_config_get_str(vtalk_vals[_i],temp,32),tret[_i]);
}
END_TEST

START_TEST(test_config_get_bool)
{
	int temp;
	int tvals[TEST_KEYS] = {0,1,0};
	ck_assert(util_config_get_bool(vtalk_keys[_i],&temp,!tvals[_i]));
	ck_assert_int_eq(temp,tvals[_i]);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Config");
	tc_single = tcase_create("Single");
	tcase_add_checked_fixture(tc_single, setup, teardown);
	tcase_add_loop_test(tc_single, test_config_get_str,0,TEST_KEYS);
	tcase_add_loop_test(tc_single, test_config_get_str_fail,0,TEST_KEYS);
	tcase_add_loop_test(tc_single, test_config_get_bool,0,TEST_KEYS);
	suite_add_tcase(s, tc_single);
	return s;
}

int main(int argc, char *argv[])
{
	Suite   *s;
	SRunner *sr;

	s  = suite_init();
	sr = srunner_create(s);

	srunner_run_all(sr, CK_NORMAL);
	srunner_free(sr);
	return 0;
}

