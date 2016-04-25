#include "utils/list.h"
#include "testing.h"
#define LIST_LENGTH 100

utils_list_s list;
utils_list_node_s* allocate_list_node()
{
	utils_list_node_s *node = malloc( sizeof(*node) );

	ck_assert(node);
	utils_list_node_init(node);
	return node;
}

void setup()
{
	ck_assert( utils_list_init(&list) );
}

void teardown() {}

START_TEST(test_list_init_destr) {}
END_TEST START_TEST(test_list_add_to_array)
{
	utils_list_node_s *nodes[LIST_LENGTH];
	utils_list_node_s *copy[LIST_LENGTH];
	utils_list_node_s *itr;
	int               c;

	for (c = 0; c < LIST_LENGTH; c++) {
		ck_assert(list.length == c);
		nodes[c] = allocate_list_node();
		utils_list_add(&list, nodes[c]);
	}
	ck_assert(list.length == LIST_LENGTH);

	c = 0;
	utils_list_for_each(list, itr) {
		ck_assert_ptr_eq(itr, nodes[LIST_LENGTH-c-1]);
		c++;
	}

	ck_assert(utils_list_to_array(&list, 0) == LIST_LENGTH);
	ck_assert(utils_list_to_array(&list,
	                              (utils_list_node_s**)&copy) ==
	          LIST_LENGTH);

	for (c = 0; c < LIST_LENGTH; c++)
		ck_assert_ptr_eq(nodes[c], copy[LIST_LENGTH-c-1]);
}

END_TEST Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("List");
	tc_single = tcase_create("Single");
	tcase_add_checked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_list_init_destr);
	tcase_add_test(tc_single, test_list_add_to_array);
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
