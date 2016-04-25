#include "testing.h"

extern void destroy_vine_talk();

void setup()
{
	destroy_vine_talk();
}

void teardown()
{

}

START_TEST(test_in_out)
{

}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Vine Talk");
	tc_single = tcase_create("Single");
	tcase_add_checked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_in_out);
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
