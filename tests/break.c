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
	utils_breakdown_stats_s * stats = malloc(sizeof(utils_breakdown_stats_s));
	utils_breakdown_instance_s * instance = malloc(sizeof(utils_breakdown_instance_s));
	char str[3] = "B@\0";
	int cnt = 0;
	utils_breakdown_init_stats(stats);
	utils_breakdown_begin(instance,stats,str);
	for(cnt = 1 ; cnt < BREAKDOWN_PARTS ; cnt++)
	{
		str[1]++;
		usleep(1000);
		utils_breakdown_advance(instance,str);
		ck_assert_int_gt(utils_breakdown_duration(instance),cnt*1000000ull);
	}
	usleep(1000);
	utils_breakdown_end(instance);
	ck_assert_int_gt(utils_breakdown_duration(instance),cnt*1000000ull);

	for(cnt = 0 ; cnt < BREAKDOWN_PARTS ; cnt++)
	{
		printf("%d: %llu\n",cnt,stats->part[cnt]);
		ck_assert_int_gt(stats->part[cnt],1000000ull);
	}
	printf("%d: %llu\n",cnt,stats->part[BREAKDOWN_PARTS]);
	free(instance);
	free(stats);
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
