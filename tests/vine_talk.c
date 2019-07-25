#include "testing.h"
#include "vine_pipe.h"
#include "core/vine_data.h"

const char config[] = "shm_file vt_test\n" "shm_size 0x100000\n";

void setup()
{
	test_backup_config();
	unlink("/dev/shm/vt_test"); /* Start fresh */

	int fd = test_open_config();

	write( fd, config, strlen(config) );

	close(fd);
}

void teardown()
{
	test_restore_config();
}

START_TEST(test_in_out) {
	vine_pipe_s *vpipe  = vine_talk_init();
	vine_pipe_s *vpipe2 = vine_talk_init();

	ck_assert(!!vpipe);
	ck_assert(!!vpipe2);
	ck_assert_ptr_eq(vpipe, vpipe2);
	vine_talk_exit();
	vine_talk_exit();
}
END_TEST

START_TEST(test_revision) {
	char * rev;
	vine_pipe_s *vpipe  = vine_talk_init();
	/**
	 * Break sha intentionally
	 */
	rev = (char*)vine_pipe_get_revision(vpipe);
	ck_assert(!!rev);
	ck_assert_str_eq(rev,VINE_TALK_GIT_REV);
	vine_talk_exit();
}
END_TEST

START_TEST(test_single_accel)
{
	int          accels;
	int          cnt;
	vine_accel   **accel_ar = 0;
	vine_accel_s *accel;
	vine_accel *vaccel,*vaccel_temp;
	vine_pipe_s  *vpipe = vine_talk_init();

	ck_assert(!!vpipe);

	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),0);

	for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
		accels = vine_accel_list(cnt, 1, 0);
		ck_assert_int_eq(accels, 0);
	}

	accel = vine_accel_acquire_type(_i);
	ck_assert(!!accel);

	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),1);
	vine_accel_release((vine_accel **)&accel);
	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),0);

	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),0);
	accel = vine_accel_init(vpipe, "FakeAccel", _i);
	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),1);

	ck_assert(!!accel);
	ck_assert_int_eq( vine_accel_get_revision(accel) ,0 );
	ck_assert_int_eq( vine_object_refs(&(accel->obj)) ,1 );

	ck_assert_ptr_eq(vine_vaccel_get_assignee(accel),0);	// Initially not assigned

	ck_assert_ptr_eq(vine_vaccel_test_set_assignee(accel,accel),accel);	// First set should work

	ck_assert_ptr_eq(vine_vaccel_test_set_assignee(accel,accel),accel);	// Same set should work

	ck_assert_ptr_eq(vine_vaccel_test_set_assignee(accel,(void*)0xBAAD),0);	// Different set should fail

	vine_accel_location(accel);

	for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
		ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),1);
		accels = vine_accel_list(cnt, 1, &accel_ar);
		if (cnt == _i || !cnt)
			ck_assert_int_eq( vine_object_refs(&(accel->obj)) ,2 );
		else
			ck_assert_int_eq( vine_object_refs(&(accel->obj)) ,1 );
		ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),1);
		if (cnt == _i || !cnt) {
			ck_assert_int_eq(accels, 1);
			if (cnt)
				ck_assert_int_eq(vine_accel_type(
				                         accel_ar[0]), cnt);
			ck_assert_ptr_eq(accel, accel_ar[0]);
			ck_assert_int_eq(vine_accel_stat(accel_ar[0],0),accel_idle);
			/* Lets get virtual! */
			ck_assert_int_eq(vine_accel_list(ANY,0,0),0);
			vaccel = accel_ar[0];

			ck_assert(vine_accel_acquire_phys(&vaccel));
			ck_assert_int_eq(vine_accel_list(ANY,0,0),1);
			ck_assert_int_eq(vine_accel_list(cnt,0,0),(cnt==_i)||(cnt==0));
			ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),1);
			ck_assert_int_eq( vine_accel_get_revision(accel) ,1+(!!cnt)*2 );
			/* got virtual accel */
			ck_assert_int_eq(((vine_accel_s*)(vaccel))->obj.type,
							 VINE_TYPE_VIRT_ACCEL);
			ck_assert(vine_vaccel_queue_size(vaccel) != -1);
			vine_vaccel_set_cid(vaccel,123);
			ck_assert_int_eq(vine_vaccel_get_cid(vaccel),123);
			vine_vaccel_set_cid(vaccel,0);
			ck_assert_int_eq(vine_vaccel_get_cid(vaccel),0);

			vine_vaccel_set_job_priority(vaccel,123);
			ck_assert_int_eq(vine_vaccel_get_job_priority(vaccel),123);
			vine_vaccel_set_job_priority(vaccel,0);
			ck_assert_int_eq(vine_vaccel_get_job_priority(vaccel),0);

			vine_vaccel_set_meta(vaccel,0);
			ck_assert_ptr_eq(vine_vaccel_get_meta(vaccel),0);
			vine_vaccel_set_meta(vaccel,(void*)0xF00F);
			ck_assert_ptr_eq(vine_vaccel_get_meta(vaccel),(void*)0xF00F);

			vine_vaccel_mark_task_done(vaccel);
			vine_vaccel_wait_task_done(vaccel);

			ck_assert(vine_vaccel_queue(((vine_vaccel_s*)(vaccel))) != 0);
			ck_assert(vine_vaccel_queue_size(((vine_vaccel_s*)(vaccel))) == 0);
			/* Cant get a virtual out of a virtual accel */
			ck_assert(!vine_accel_acquire_phys(&vaccel));
			ck_assert_int_eq(vine_accel_stat(vaccel,0),accel_idle);
			vine_accel_location(vaccel);

			// Should not be reclaimable yet
			vaccel_temp = vaccel;
			vine_accel_release(&(vaccel_temp));
			ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),1);
			vaccel_temp = vaccel;
			vine_accel_release(&(vaccel_temp));
			ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),0);
			ck_assert_int_eq( vine_accel_get_revision(accel) ,2+(!!cnt)*2 );
		} else {
			ck_assert_int_eq(accels, 0);
		}
		if (cnt == _i || !cnt)
		{
			ck_assert_int_eq( vine_object_refs(&(accel->obj)) ,2 );
		}
	}
	vine_accel_list_free(accel_ar);
	ck_assert_int_eq( vine_object_refs(&(accel->obj)) ,1 );
	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),1);
	ck_assert( !vine_pipe_delete_accel(vpipe, accel) );
	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),0);
	ck_assert( vine_pipe_delete_accel(vpipe, accel) );
	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL),0);

	vine_talk_exit();
	/* setup()/teardown() */
}
END_TEST

vine_proc_s * create_proc(vine_pipe_s * vpipe, int type, const char * name,void * pd,size_t psize)
{
	vine_proc_s * proc;
	ck_assert(!!vpipe);
	ck_assert( !vine_proc_get(type, "TEST_PROC") );
	proc = (vine_proc_s*)vine_proc_register(type,"TEST_PROC",pd,psize);
	return proc;
}

START_TEST(test_single_proc)
{
	int         cnt;
	size_t      cs;
	vine_proc_s *proc;
	vine_pipe_s *vpipe = vine_talk_init();
	char        pd[]   = "TEST_DATA";

	ck_assert(!!vpipe);

	proc = create_proc(vpipe,_i,"TEST_PROC",pd,_i);

	if(_i)
		ck_assert( !!proc );
	else /* Fail to create an ANY procedure */
	{
		ck_assert( !proc );
		ck_assert( !vine_proc_get(_i, "TEST_PROC") );
		vine_talk_exit();
		return;
	}

	ck_assert_ptr_eq( vine_proc_register(_i,"TEST_PROC","DIFF_DATA",_i) , 0 );

	ck_assert( vine_proc_get_code(proc, &cs) != NULL);
	ck_assert_int_eq(cs, _i);
	ck_assert( vine_proc_match_code(proc, pd, _i) );
	ck_assert( !vine_proc_match_code(proc, pd, _i-1) );

	VineFunctor * vf = vine_proc_get_functor(proc);

	ck_assert( strncmp((char*)&vf, pd, _i) == 0);

	for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
		if (cnt == _i || cnt == ANY)
			ck_assert( vine_proc_get(cnt, "TEST_PROC") != NULL);
		else
			ck_assert( !vine_proc_get(cnt, "TEST_PROC") );
	}

	vine_proc_put(proc);

	ck_assert_int_eq(vine_pipe_delete_proc(vpipe,proc),0);
	ck_assert_int_eq(vine_pipe_delete_proc(vpipe,proc),1);

	vine_talk_exit();
}
END_TEST

START_TEST(test_alloc_data)
{
	vine_pipe_s *vpipe = vine_talk_init();
	size_t size = _i;
	ck_assert(!!vpipe);
	vine_data * data = vine_data_init(vpipe,0,size);

	vine_data_check_flags(data);

	ck_assert(data != NULL);

	ck_assert(vine_data_deref(data) != NULL);

	ck_assert_ptr_eq(vine_data_ref(vine_data_deref(data)),data);

	ck_assert_int_eq(vine_data_size(data),size);

	ck_assert(!vine_data_check_ready(vpipe, data));
	vine_data_mark_ready(vpipe, data);
	ck_assert(vine_data_check_ready(vpipe, data));

	// Just call these functions - they should not crash
	// Eventually add more thorough tests.
	vine_data_arg_init(data,0);
	vine_data_input_init(data,0);
	vine_data_output_init(data,0);
	vine_data_output_done(data);
	vine_data_memcpy(0,data,data,0);

	// vine_data_sync_to_remote should be a no-op when data are in remote

	vine_data_modified(data,REMT_SYNC);
	vine_data_sync_to_remote(0,data,0);

	// vine_data_sync_from_remote should be a no-op when data are in USR

	vine_data_modified(data,USER_SYNC);
	vine_data_sync_from_remote(0,data,0);

	// vine_data_sync_from_remote should be a no-op when data are in SHM (and user pointer is null)

	vine_data_modified(data,SHM_SYNC);
	vine_data_sync_from_remote(0,data,0);

	vine_data_free(data);

	vine_talk_exit();
}
END_TEST

START_TEST(test_alloc_data_alligned)
{
	vine_pipe_s *vpipe = vine_talk_init();
	size_t size = _i&256;
	size_t align = 1<<(_i/256);
	ck_assert(!!vpipe);

	vine_data * data = vine_data_init_aligned(vpipe,0,size,align);

	vine_data_stat(data);

	vine_data_check_flags(data);

	ck_assert(data != NULL);

	ck_assert(vine_data_deref(data) != NULL);

	ck_assert((size_t)vine_data_deref(data) % align == 0);

	ck_assert_ptr_eq(vine_data_ref(vine_data_deref(data)),data);

	ck_assert_int_eq(vine_data_size(data),size);

	ck_assert(!vine_data_check_ready(vpipe, data));
	vine_data_mark_ready(vpipe, data);
	ck_assert(vine_data_check_ready(vpipe, data));

	vine_data_free(data);

	ck_assert_ptr_eq(vine_data_init_aligned(vpipe,0,size,0),0);

	vine_talk_exit();
}
END_TEST

START_TEST(test_task_issue_and_wait_v1)
{
	vine_pipe_s *vpipe = vine_talk_init();
	vine_vaccel_s * accel;
	vine_accel_type_e at = _i%VINE_ACCEL_TYPES;
	ck_assert(!!vpipe);

	vine_proc_s *proc = create_proc(vpipe,at,"issue_proc",0,0);

	if(!at)
	{
		ck_assert( !proc );
		ck_assert( !vine_proc_get(at, "TEST_PROC") );
		vine_talk_exit();
		return;
	}

	accel = vine_accel_acquire_type(at);

	vine_task * task = vine_task_issue(accel,proc,0,0,0,0,0,0);

	vine_pipe_wait_for_task(vpipe,at);

	ck_assert_int_eq(vine_task_stat(task,0),task_issued);

	vine_task_mark_done(task,task_completed);

	ck_assert_int_eq(vine_task_stat(task,0),task_completed);

	vine_task_wait_done(task);

	vine_task_free(task);

	vine_proc_put(proc);

	vine_talk_exit();
}
END_TEST

START_TEST(test_task_issue_and_wait_v2)
{
	vine_pipe_s *vpipe = vine_talk_init();
	vine_vaccel_s * accel;
	vine_accel_type_e at = _i%VINE_ACCEL_TYPES;
	vine_accel_type_e wait_type = at*(_i/VINE_ACCEL_TYPES);
	ck_assert(!!vpipe);
	vine_task_stats_s stats;

	vine_proc_s *proc = create_proc(vpipe,at,"issue_proc",0,0);

	if(!at)
	{
		ck_assert( !proc );
		ck_assert( !vine_proc_get(at, "TEST_PROC") );
		vine_talk_exit();
		return;
	}

	accel = vine_accel_acquire_type(wait_type);

	vine_task * task = vine_task_issue(accel,proc,0,0,0,0,0,0);

	vine_pipe_wait_for_task_type_or_any_assignee(vpipe,at,0);

	ck_assert_int_eq(vine_task_stat(task,&stats),task_issued);

	vine_task_mark_done(task,task_completed);

	ck_assert_int_eq(vine_task_stat(task,&stats),task_completed);

	vine_task_wait(task);

	vine_task_free(task);

	vine_proc_put(proc);

	vine_talk_exit();
}
END_TEST

START_TEST(test_type_strings)
{
	switch(_i)
	{
		case 0 ... VINE_ACCEL_TYPES:
			ck_assert_int_eq(_i,vine_accel_type_from_str(vine_accel_type_to_str(_i)));
			break;
		case VINE_ACCEL_TYPES+1:
			ck_assert_int_eq(vine_accel_type_from_str("NotRealyAType"),VINE_ACCEL_TYPES);
			ck_assert(!vine_accel_type_to_str(VINE_ACCEL_TYPES));
			ck_assert(!vine_accel_type_to_str(VINE_ACCEL_TYPES+1));
			break;
	}
}
END_TEST

START_TEST(test_empty_task)
{
	vine_pipe_s *vpipe  = vine_talk_init();

	vine_task * task = vine_task_alloc(vpipe,0,0);

	vine_task_mark_done(task,task_completed);

	vine_task_wait(task);

	vine_task_free(task);

	vine_talk_exit();
}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_single;

	s         = suite_create("Vine Talk");
	tc_single = tcase_create("Single");
	tcase_add_unchecked_fixture(tc_single, setup, teardown);
	tcase_add_loop_test(tc_single, test_in_out,0,10);
	tcase_add_test(tc_single, test_revision);
	tcase_add_loop_test(tc_single, test_single_accel, 0, VINE_ACCEL_TYPES);
	tcase_add_loop_test(tc_single, test_single_proc, 0, VINE_ACCEL_TYPES);
	tcase_add_loop_test(tc_single, test_alloc_data, 0, 1024);
	tcase_add_loop_test(tc_single, test_alloc_data_alligned, 0, 4096);
	tcase_add_loop_test(tc_single,test_task_issue_and_wait_v1,0,VINE_ACCEL_TYPES);
	tcase_add_loop_test(tc_single,test_task_issue_and_wait_v2,0,VINE_ACCEL_TYPES*2);
	tcase_add_loop_test(tc_single, test_type_strings, 0, VINE_ACCEL_TYPES+2);
	tcase_add_test(tc_single, test_empty_task);
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
