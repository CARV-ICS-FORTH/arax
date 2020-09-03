#include "utils/Kv.h"
#include "testing.h"
#define TEST_LENGTH VINE_KV_CAP

utils_kv_s kv;

void setup()
{
	test_common_setup();
	utils_kv_init(&kv);
	ck_assert( kv.pairs==0);
}

void teardown()
{
	test_common_teardown();
}


START_TEST(test_get)
{
	ck_assert_ptr_eq(utils_kv_get(&kv,0),0);
}
END_TEST

START_TEST(test_set)
{
	ck_assert_ptr_eq(utils_kv_get(&kv,0),0);
	utils_kv_set(&kv,0,(void*)1);
	ck_assert(utils_kv_get(&kv,0));
	ck_assert_ptr_eq(*utils_kv_get(&kv,0),(void*)1);
	utils_kv_set(&kv,0,(void*)2);
	ck_assert(utils_kv_get(&kv,0));
	ck_assert_ptr_eq(*utils_kv_get(&kv,0),(void*)2);
	utils_kv_set(&kv,0,(void*)0);
	ck_assert(utils_kv_get(&kv,0));
	ck_assert_ptr_eq(*utils_kv_get(&kv,0),(void*)0);
}
END_TEST

START_TEST(test_get_set)
{
	size_t cnt;

	for(cnt = 0 ; cnt < TEST_LENGTH ; cnt++)
	{
		ck_assert_ptr_eq(utils_kv_get(&kv,(void*)cnt),0);
		utils_kv_set(&kv,(void*)cnt,(void*)cnt);
		ck_assert_ptr_eq(*utils_kv_get(&kv,(void*)cnt),(void*)cnt);
	}

	for(cnt = 0 ; cnt < TEST_LENGTH ; cnt++)
	{
		ck_assert_ptr_eq(*utils_kv_get(&kv,(void*)cnt),(void*)cnt);
	}
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Kv");
	tc_single = tcase_create("Single");
	tcase_add_checked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_get);
	tcase_add_test(tc_single, test_set);
	tcase_add_test(tc_single, test_get_set);
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
