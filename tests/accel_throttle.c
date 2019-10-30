#include "testing.h"
#include "vine_pipe.h"      //init test
#include "core/vine_data.h" //free date
#include "core/vine_accel.h"//inc dec size

#define GPU_SIZE    2097152
#define DATA_SIZE   1097152
#define BIG_SIZE    1597152

async_meta_s meta;

const char config[] = "shm_file vt_test\n" "shm_size 0x100000\n";

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

void* size_inc(void* accel)
{
	vine_accel_size_inc(accel,DATA_SIZE);
    return 0;
}

void* size_big_inc(void* accel)
{
	vine_accel_size_inc(accel,BIG_SIZE);
    return 0;
}

void* size_big_dec(void* accel)
{
	vine_accel_size_dec(accel,BIG_SIZE);
    return 0;
}

void* size_dec(void* accel)
{
	vine_accel_size_dec(accel,DATA_SIZE);
    return 0;
}

vine_proc_s * create_proc(vine_pipe_s * vpipe, int type, const char * name,void * pd,size_t psize)
{
	vine_proc_s * proc;
	ck_assert(!!vpipe);
	ck_assert( !vine_proc_get(type, "TEST_PROC") );
	proc = (vine_proc_s*)vine_proc_register(type,"TEST_PROC",pd,psize);
	return proc;
}

START_TEST(test_gpu_size)
{
    vine_accel_s* accel;
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //init accel
    accel = vine_accel_init(mypipe, "FakeAccel", 1 , GPU_SIZE);
    ck_assert(accel!=0);
    ck_assert_int_eq(accel->AvaliableSize,GPU_SIZE);
    //releaseAccelerator
    ck_assert_int_eq(vine_object_refs((vine_object_s*)accel),1);
	vine_accel_release((vine_accel **)&accel);
    ck_assert_ptr_eq(accel,0);
    
    //exit vine_talk
	vine_talk_exit();
}
END_TEST

START_TEST(test_thread_inc_dec_size_simple)
{
    //staff to use
    size_t size_defore;
    vine_accel_s *accel,*myaccel;
    vine_accel_type_e accelType = GPU; //GPU : 1
	
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //create proc
	vine_proc_s *process_id = create_proc(mypipe,accelType,"issue_proc",0,0);
    ck_assert( !!process_id );

    //initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE);
    ck_assert(accel!=0);
    ck_assert_int_eq(accel->AvaliableSize,GPU_SIZE);
    size_defore = GPU_SIZE;
    
    //acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
	ck_assert(!!myaccel);
    
    vine_accel_set_physical(myaccel,accel);
    
    pthread_t * thread;
    
    //test inc
    thread = spawn_thread(size_inc,myaccel);
	wait_thread(thread);
    ck_assert_int_eq( vine_accel_get_AvaliableSize(accel) ,size_defore+DATA_SIZE);
    ck_assert_int_eq( vine_accel_get_size(myaccel)        ,size_defore+DATA_SIZE);
    
    //test dec
    thread = spawn_thread(size_dec,myaccel);
	wait_thread(thread);
    ck_assert_int_eq(accel->AvaliableSize,size_defore);
    
    //exit vine_talk
	vine_talk_exit();
}
END_TEST

START_TEST(test_thread_wait)
{
    vine_accel_s *accel,*myaccel;
    vine_accel_type_e accelType = GPU; //GPU : 1
	
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //create proc
	vine_proc_s *process_id = create_proc(mypipe,accelType,"issue_proc",0,0);
    ck_assert( !!process_id );

    //initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE);
    ck_assert(accel!=0);
    ck_assert_int_eq(accel->AvaliableSize,GPU_SIZE);
    
    //acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
	ck_assert(!!myaccel);
    
    vine_accel_set_physical(myaccel,accel);
    
    pthread_t * thread1,* thread2,*thread3,*thread4;
    
    thread1 = spawn_thread(size_big_dec,myaccel);
    wait_thread(thread1);
    thread1 = spawn_thread(size_big_dec,myaccel);
    thread3 = spawn_thread(size_big_dec,myaccel);
    usleep(1000);
    thread2 = spawn_thread(size_big_inc,myaccel);
    usleep(1000);
    thread4 = spawn_thread(size_big_inc,myaccel);
    usleep(1000);
    
    wait_thread(thread4);
    wait_thread(thread3);
    wait_thread(thread2);
    wait_thread(thread1);
    //exit vine_talk
	vine_talk_exit();
}
END_TEST

START_TEST(test_single_phys_task_issue)
{
    //staff to use
	vine_accel_s *myaccel,*accel;
    vine_accel_type_e accelType = 1; //GPU
	
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //create proc
	vine_proc_s *process_id = create_proc(mypipe,accelType,"issue_proc",0,0);
    ck_assert( !!process_id );

    //initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE);
    ck_assert(accel!=0);
    
    //acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
	ck_assert(!!myaccel);
     
    //make datain
    //vine_buffer_s datain[1] = {VINE_BUFFER(null,1048576)};
    //vine_data_modified(datain[0], USER_SYNC);
    //printf("\tSync to inputs\n");
    //vine_data_sync_to_remote(accelInUse,inputs[0],true);
    
    //init vine_task to add size 
    ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),1);
	vine_task * task = vine_task_issue(myaccel,process_id,0,0,0,0,0,0);
	ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),2);
    //wait for task to come for exec.
	vine_pipe_wait_for_task(mypipe,accelType);
	ck_assert_int_eq(vine_task_stat(task,0),task_issued);
    
    //mark done after exec
	vine_task_mark_done(task,task_completed);
	ck_assert_int_eq(vine_task_stat(task,0),task_completed);
	vine_task_wait_done(task);
    
    

    //delete task
    ck_assert_int_eq(vine_object_refs((vine_object_s*)task),1);
	vine_task_free(task);
    
    //vine_data_free(inputs[0]);
    
    //releaseAccelerator check ref_count check free done
	ck_assert_int_eq(get_object_count(&(mypipe->objs),VINE_TYPE_VIRT_ACCEL),1);
    ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),1);
	vine_accel_release((vine_accel **)&myaccel);
    ck_assert_ptr_eq(myaccel,0);
	
    //exit vine_talk
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
	//add tests here
    tcase_add_test(tc_single, test_gpu_size);
    tcase_add_test(tc_single, test_thread_inc_dec_size_simple);
    tcase_add_test(tc_single, test_single_phys_task_issue);
    tcase_add_loop_test(tc_single, test_thread_wait ,0 ,10);
    //
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
	srunner_set_fork_status(sr, CK_NOFORK);//To debug CK_NOFORK
	srunner_run_all(sr, CK_NORMAL);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
