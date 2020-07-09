#include "testing.h"
#include "vine_pipe.h"      //init test
#include <pthread.h>

async_meta_s meta;

const char config[] = "shm_file vt_test\n" "shm_size 10000000\n";

void setup()
{
	test_backup_config();
	unlink("/dev/shm/vt_test"); /* Start fresh */

	int fd = test_open_config();

	write( fd, config, strlen(config) );

	close(fd);

    // This will not work for ivshmem
	async_meta_init_once(&meta,0);
}

void teardown()
{
	test_restore_config();
    async_meta_exit(&meta);
}

START_TEST(test_init)
{
	//init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);

    ///Check
    ck_assert_int_eq( vine_pipe_get_available_size(mypipe), 10000000);
    ck_assert_int_eq( vine_pipe_get_total_size(mypipe), 10000000);

    //exit vine_talk
	vine_talk_exit();
	return ;
}
END_TEST

START_TEST(test_inc_dec)
{
	//init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);

    ///Check init
    ck_assert_int_eq( vine_pipe_get_available_size(mypipe), 10000000);
    ck_assert_int_eq( vine_pipe_get_total_size(mypipe), 10000000);
	//check dec
	if(_i>0)
		vine_pipe_size_dec(mypipe,_i);
	ck_assert_int_eq(vine_pipe_get_available_size(mypipe), (10000000)-_i);

	//check inc
	if(_i>0)
		vine_pipe_size_inc(mypipe,_i);
	ck_assert_int_eq(vine_pipe_get_available_size(mypipe), 10000000);

	//exit vine_talk
	vine_talk_exit();
	return ;
}
END_TEST

void* size_inc(void* pipe)
{
	vine_pipe_size_inc((vine_pipe_s*)pipe,3000000);
    return 0;
}

void* size_dec(void* pipe)
{
	vine_pipe_size_dec((vine_pipe_s*)pipe,3000000);
    return 0;
}

void* size_dec_big(void* pipe)
{
	vine_pipe_size_dec((vine_pipe_s*)pipe, 8000000);
    return 0;
}

START_TEST(test_wait)
{
	//initialize
	pthread_t *thread1,* thread2,*thread3,*thread4;

    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);

    ///Check
    ck_assert_int_eq( vine_pipe_get_available_size(mypipe), 10000000);
    ck_assert_int_eq( vine_pipe_get_total_size(mypipe), 10000000);

	thread1 = spawn_thread(size_dec_big,mypipe);
	wait_thread(thread1);
    ck_assert_int_eq(vine_pipe_get_available_size(mypipe),2000000);

	thread1 = spawn_thread(size_dec,mypipe);
    thread2 = spawn_thread(size_dec,mypipe);
	safe_usleep(1000);
	ck_assert_int_eq(vine_pipe_get_available_size(mypipe),2000000);

	thread3 = spawn_thread(size_inc,mypipe);
	safe_usleep(1000);
	ck_assert_int_eq(vine_pipe_get_available_size(mypipe),2000000);

	thread4 = spawn_thread(size_inc,mypipe);
	safe_usleep(1000);
	ck_assert_int_eq(vine_pipe_get_available_size(mypipe),2000000);

	wait_thread(thread4);
	wait_thread(thread3);
    wait_thread(thread2);
	wait_thread(thread1);

	ck_assert_int_eq(vine_pipe_get_available_size(mypipe), 2000000);

	vine_pipe_size_inc(mypipe,8000000);

	//exit vine_talk
	vine_talk_exit();
	return ;
}
END_TEST

START_TEST(test_assert_dec)
{
	vine_pipe_size_dec(0,3000000);
}
END_TEST

START_TEST(test_assert_inc)
{
	vine_pipe_size_inc(0,3000000);
}
END_TEST

START_TEST(test_assert_get_1)
{
	vine_pipe_get_available_size(0);
}
END_TEST

START_TEST(test_assert_get_2)
{
	vine_pipe_get_total_size(0);
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Pipe Throttle");
	tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
	//add tests here
    tcase_add_test(tc_single, test_init);
	tcase_add_test(tc_single, test_inc_dec);
    tcase_add_test(tc_single, test_wait);
	//now check asserts
	tcase_add_test_raise_signal(tc_single, test_assert_dec,6);
	tcase_add_test_raise_signal(tc_single, test_assert_inc,6);
	tcase_add_test_raise_signal(tc_single, test_assert_get_1,6);
	tcase_add_test_raise_signal(tc_single, test_assert_get_2,6);
	//end of tests
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
	srunner_set_fork_status(sr, CK_FORK);//To debug CK_NOFORK
	srunner_run_all(sr, CK_NORMAL);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
