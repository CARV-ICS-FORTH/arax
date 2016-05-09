#include "testing.h"
#include "vine_pipe.h"

const char config[] = "shm_file vt_test\n" "shm_size 0x10000\n";

void setup()
{
	test_backup_config();
	unlink("/dev/shm/vt_test"); /* Start fresh */

	int fd = test_open_config();

	write( fd, config, strlen(config) );
}

void teardown()
{
	test_restore_config();
}

START_TEST(test_in_out) {
	/* setup()/teardown()*/
}
END_TEST

START_TEST(test_single_accel)
{
	int          accels;
	int          cnt;
	vine_accel   **accel_ar;
	vine_accel_s *accel;
	vine_pipe_s  *vpipe = vine_pipe_get();

	ck_assert(vpipe);

	for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
		accels = vine_accel_list(cnt, 0);
		ck_assert_int_eq(accels, 0);
	}

	accel =
	        arch_alloc_allocate( vpipe->allocator,
	                             vine_accel_calc_size("FakeAccel") );

	ck_assert(accel);

	accel = vine_accel_init(accel, "FakeAccel", _i);

	ck_assert(accel);

	vine_pipe_register_accel(vpipe, accel);

	for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
		accels = vine_accel_list(cnt, &accel_ar);
		if (cnt == _i || !cnt) {
			ck_assert_int_eq(accels, 1);
			ck_assert_ptr_eq(accel, accel_ar[0]);
			if (cnt)
				ck_assert_int_eq(vine_accel_type(
				                         accel_ar[0]), cnt);
		} else {
			ck_assert_int_eq(accels, 0);
		}
	}

	ck_assert( !vine_pipe_delete_accel(vpipe, accel) );

	/* setup()/teardown()*/
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Vine Talk");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_test(tc_single, test_in_out);
	tcase_add_loop_test(tc_single, test_single_accel, 0, VINE_ACCEL_TYPES);
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
