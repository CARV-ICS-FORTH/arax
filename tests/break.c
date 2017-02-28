#include "testing.h"
#include "utils/breakdown.h"

#ifndef BREAKS_ENABLE
int main(int argc, char *argv[])
{
	return EXIT_SUCCESS;
}
#else

void setup()
{
}

void teardown()
{
}

START_TEST(duration_check)
{
	utils_breakdown_stats_s stats;
	utils_breakdown_instance_s instance;

	utils_breakdown_init_stats(&stats);
	utils_breakdown_begin(&instance,&stats,"A");
	usleep(1000);
	utils_breakdown_advance(&instance,"B");
	printf("Duration: %llu\n",utils_breakdown_duration(&instance));
	ck_assert(utils_breakdown_duration(&instance) > 900000);
	ck_assert(utils_breakdown_duration(&instance) < 1100000);
	usleep(1000);
	utils_breakdown_advance(&instance,"C");
	printf("Duration: %llu\n",utils_breakdown_duration(&instance));
	ck_assert(utils_breakdown_duration(&instance) > 1100000);
	ck_assert(utils_breakdown_duration(&instance) < 2200000);
	utils_breakdown_advance(&instance,"D");
	printf("Duration: %llu\n",utils_breakdown_duration(&instance));
	ck_assert(utils_breakdown_duration(&instance) > 1100000);
	ck_assert(utils_breakdown_duration(&instance) < 2200000);
	usleep(1000);
	utils_breakdown_end(&instance);
	printf("Duration: %llu\n",utils_breakdown_duration(&instance));
	ck_assert(utils_breakdown_duration(&instance) > 2200000);
	ck_assert(utils_breakdown_duration(&instance) < 3300000);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Breakdown");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, duration_check);
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
	srunner_set_fork_status(sr, CK_NOFORK);
	srunner_run_all(sr, CK_NORMAL);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
#endif
