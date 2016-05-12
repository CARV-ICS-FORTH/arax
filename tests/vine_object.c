#include "core/vine_object.h"
#include "testing.h"

vine_object_repo_s repo;

void setup()
{
	vine_object_repo_init(&repo);
}

void teardown()
{
	vine_object_repo_exit(&repo);
}

START_TEST(test_vine_object_init_destr) {}
END_TEST

START_TEST(test_vine_object_leak)
{
	vine_object_s obj;
	vine_object_register(&repo,&obj,_i,"Obj");
	ck_assert_int_eq(vine_object_repo_exit(&repo),1);
	vine_object_remove(&repo,&obj);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Queue");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_vine_object_init_destr);
	tcase_add_loop_test(tc_single, test_vine_object_leak,0,VINE_TYPE_COUNT);
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
