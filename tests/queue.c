#include <stddef.h>
#include <stdint.h>

#include "utils/queue.h"
#include "testing.h"

#define BUFF_SIZE (sizeof(utils_queue_s)+2)

char          buff[BUFF_SIZE];
utils_queue_s *queue;
char FULL_OF_FF = 0xFF;

void setup()
{
	memset(buff,FULL_OF_FF,BUFF_SIZE);
	queue = utils_queue_init(buff+1);
	ck_assert(queue);
	ck_assert( !utils_queue_used_slots(queue) );
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
	int c = UTILS_QUEUE_CAPACITY;

	while (c) {
		ck_assert( utils_queue_push(queue, (void*)(size_t)c) );
		c--;
	}

	ck_assert( !utils_queue_push(queue, (void*)(size_t)c+1) );

	c = utils_queue_used_slots(queue);
	while (c) {
		ck_assert_ptr_eq(utils_queue_pop(queue), (void*)(size_t)c);
		c--;
	}

	ck_assert( !utils_queue_used_slots(queue) );
	ck_assert_ptr_eq(utils_queue_pop(queue), 0);
}
END_TEST

/* Check that the circular buffer works OK */
START_TEST(test_queue_circulation)
{
	/* Fill half buffer */
	int c = UTILS_QUEUE_CAPACITY/2;

	while (c) {
		ck_assert( utils_queue_push(queue, (void*)(size_t)c) );
		c--;
	}

	/* Empty it */
	c = utils_queue_used_slots(queue);
	while (c) {
		ck_assert_ptr_eq(utils_queue_pop(queue), (void*)(size_t)c);
		c--;
	}

	/* Now fill it and it should wrap around */
	c = UTILS_QUEUE_CAPACITY;
	while (c) {
		ck_assert( utils_queue_push(queue, (void*)(size_t)c) );
		c--;
	}

	ck_assert( !utils_queue_push(queue, (void*)(size_t)c+1) );

	c = utils_queue_used_slots(queue);
	while (c) {
		ck_assert_ptr_eq(utils_queue_pop(queue), (void*)(size_t)c);
		c--;
	}

	ck_assert( !utils_queue_used_slots(queue) );
	ck_assert_ptr_eq(utils_queue_pop(queue), 0);
}
END_TEST

/* Check that uint16_t wrap around does not cause any issues */
START_TEST(test_queue_indices_circulation)
{
	int c = UTILS_QUEUE_CAPACITY;

	while (c) {
		ck_assert( utils_queue_push(queue, (void*)(size_t)c) );
		c--;
	}

	ck_assert_ptr_eq(utils_queue_push(queue, (void*)0xFF), 0);
	ck_assert_int_eq(utils_queue_used_slots(queue), UTILS_QUEUE_CAPACITY);

	c = 2*UINT16_MAX;
	while (c) {
		ck_assert(utils_queue_pop(queue));
		ck_assert_ptr_eq(utils_queue_push(queue, (void*)(size_t)c), (void*)(size_t)c);
		/* ck_assert(!utils_queue_push(queue, (void*)(size_t)c+1)); */
		c--;
	}

	ck_assert_ptr_eq(utils_queue_push(queue, (void*)0xFF), 0);
	ck_assert_int_eq(utils_queue_used_slots(queue), UTILS_QUEUE_CAPACITY);
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
	tcase_add_test(tc_single, test_queue_circulation);
	tcase_add_test(tc_single, test_queue_indices_circulation);
	/* tcase_set_timeout(tc_single,900); */
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
