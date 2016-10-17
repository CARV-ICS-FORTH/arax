#include "async.h"
#include "testing.h"

async_meta_s meta;

void setup()
{
	async_meta_init_once(&meta);
}

void teardown()
{
	async_meta_exit(&meta);
}

START_TEST(serial_completion)
{
	async_completion_s completion;
	async_completion_init(&meta,&completion);
	ck_assert(!async_completion_check(&meta,&completion));
	ck_assert(!async_completion_check(&meta,&completion));
	async_completion_complete(&meta,&completion);
	ck_assert(async_completion_check(&meta,&completion));
	ck_assert(async_completion_check(&meta,&completion));
	async_completion_wait(&meta,&completion);
}
END_TEST

START_TEST(serial_semaphore)
{
	async_semaphore_s sem;
	async_semaphore_init(&meta,&sem);
	ck_assert_int_eq(async_semaphore_value(&meta,&sem),0);
	async_semaphore_inc(&meta,&sem);
	ck_assert_int_eq(async_semaphore_value(&meta,&sem),1);
	async_semaphore_dec(&meta,&sem);
	ck_assert_int_eq(async_semaphore_value(&meta,&sem),0);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Async");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, serial_completion);
	tcase_add_test(tc_single, serial_semaphore);
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
