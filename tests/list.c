#include "utils/list.h"
#include "testing.h"
#define TEST_LENGTH 100

utils_list_s list;
utils_list_node_s* allocate_list_node()
{
	utils_list_node_s *node = malloc( sizeof(*node) );

	ck_assert(node);
	utils_list_node_init(node);
	return node;
}

void free_list_node(utils_list_node_s *node)
{
	ck_assert(node);
	free(node);
}

void setup()
{
	ck_assert( utils_list_init(&list) );
}

void teardown() {}

START_TEST(test_list_init_destr) {}
END_TEST

START_TEST(test_list_add_del_to_array)
{
	utils_list_node_s **nodes;
	utils_list_node_s **copy;
	utils_list_node_s *itr;
	int               c;

	nodes = malloc(sizeof(utils_list_node_s *)*_i);
	copy = malloc(sizeof(utils_list_node_s *)*_i);

	for (c = 0; c < _i; c++) {
		ck_assert(list.length == c);
		nodes[c] = allocate_list_node();
		utils_list_add(&list, nodes[c]);
	}
	ck_assert(list.length == _i);

	c = 0;
	utils_list_for_each(list, itr) {
		ck_assert_ptr_eq(itr, nodes[_i-1-c]);
		ck_assert_int_lt(c,_i);
		c++;
	}
	c = 0;
	utils_list_for_each_reverse(list, itr) {
		ck_assert_ptr_eq(itr, nodes[c]);
		ck_assert_int_lt(c,_i);
		c++;
	}

	ck_assert(utils_list_to_array(&list, 0) == _i);
	ck_assert(utils_list_to_array(&list,
	                              (utils_list_node_s**)copy) ==
	          _i);

	for (c = 0; c < _i; c++)
		ck_assert_ptr_eq(nodes[c], copy[_i-c-1]);

	while (list.length) {
		free_list_node( utils_list_del(&list, list.head.next) );
	}
	ck_assert_int_eq(list.length, 0);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("List");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_list_init_destr);
	tcase_add_loop_test(tc_single, test_list_add_del_to_array,0,TEST_LENGTH);
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
