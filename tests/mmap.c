#include "testing.h"
#include <sys/mman.h>
void setup()
{
}

void teardown()
{
}

START_TEST(forced)
{
	unsigned int * map1 = mmap(0,4096,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_SHARED |MAP_ANON,-1,0);
	ck_assert_ptr_ne(map1,MAP_FAILED);
	memset(map1,0xFE,4096);
	ck_assert_int_eq(*map1,0xFEFEFEFE);
	unsigned int * map2 = mmap(map1,4096,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_SHARED |MAP_ANON| MAP_FIXED,-1,0);
	ck_assert_ptr_ne(map2,MAP_FAILED);
	ck_assert_int_ne(*map2,0xFE);
	ck_assert_int_eq(*map2,0);
}
END_TEST

START_TEST(lax)
{
	unsigned int * map1 = mmap(0,4096,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_SHARED |MAP_ANON,-1,0);
	ck_assert_ptr_ne(map1,MAP_FAILED);
	memset(map1,0xFE,4096);
	ck_assert_int_eq(*map1,0xFEFEFEFE);
	ck_assert_int_ne(*map1,0);

	unsigned int * map2 = mmap(map1,4096,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_SHARED |MAP_ANON,-1,0);
	ck_assert_ptr_ne(map2,MAP_FAILED);
	ck_assert_ptr_ne(map2,map1);
	ck_assert_int_ne(*map2,0xFEFEFEFE);
	ck_assert_int_eq(*map2,0);
	ck_assert_int_eq(*map1,0xFEFEFEFE);
	ck_assert_int_ne(*map1,0);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("mmap");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, forced);
	tcase_add_test(tc_single, lax);
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
