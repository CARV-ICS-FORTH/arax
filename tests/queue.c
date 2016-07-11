#include "utils/queue.h"
#include "testing.h"

#define BUFF_SIZE 1024

char          buff[BUFF_SIZE];
utils_queue_s *queue;
char FULL_OF_FF = 0xFF;

void setup()
{
	memset(buff,FULL_OF_FF,BUFF_SIZE);
	queue = utils_queue_init(buff+1, BUFF_SIZE-2);
	ck_assert(queue);
	ck_assert( !utils_queue_used_slots(queue) );
	ck_assert( utils_queue_free_slots(queue) );
}

void teardown()
{
	ck_assert_int_eq(buff[0],FULL_OF_FF);
	ck_assert_int_eq(buff[BUFF_SIZE-1],FULL_OF_FF);
}

START_TEST(test_queue_init_destr)
{}
END_TEST

START_TEST(test_queue_push_pop)
{
	int c = utils_queue_free_slots(queue);

	while (c) {
		ck_assert( utils_queue_push(queue, (void*)(size_t)c) );
		c--;
	}

	ck_assert( !utils_queue_push(queue, (void*)(size_t)c) );

	c = utils_queue_used_slots(queue);
	while (c) {
		ck_assert_ptr_eq(utils_queue_pop(queue), (void*)(size_t)c);
		c--;
	}

	ck_assert_ptr_eq(utils_queue_pop(queue), 0);
}
END_TEST

START_TEST(test_fit)
{
	if(_i < utils_queue_calc_bytes(1))
		ck_assert(!utils_queue_init(buff+1,_i ));
	else
	{
		ck_assert(utils_queue_init(buff+1,_i ));
		ck_assert_int_le
		(utils_queue_calc_bytes(utils_queue_used_slots(queue)),_i);
	}
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;
	s         = suite_create("Queue");
	tc_single = tcase_create("Single");
	tcase_add_checked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_queue_init_destr);
	tcase_add_test(tc_single, test_queue_push_pop);
	tcase_add_loop_test(tc_single, test_fit,0,BUFF_SIZE*2);
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
