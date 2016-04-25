#include "utils/queue.h"
#include "testing.h"

char          buff[4096];
utils_queue_s *queue;

void setup()
{
	queue = utils_queue_init(buff, 4096);
	ck_assert(queue);
	ck_assert( !utils_queue_used_slots(queue) );
	ck_assert( utils_queue_free_slots(queue) );
}

void teardown() {}

START_TEST(test_queue_init_destr) {}
END_TEST START_TEST(test_queue_push_pop)
{
	int c = utils_queue_free_slots(queue);

	while (c) {
		ck_assert( utils_queue_push(queue, (void*)(size_t)c) );
		c--;
	}
	c = utils_queue_free_slots(queue);
	while (c) {
		ck_assert_ptr_eq(utils_queue_pop(queue), (void*)(size_t)c);
		c--;
	}
}

END_TEST Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Queue");
	tc_single = tcase_create("Single");
	tcase_add_checked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_queue_init_destr);
	tcase_add_test(tc_single, test_queue_push_pop);
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
