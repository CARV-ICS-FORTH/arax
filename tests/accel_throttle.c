#include "testing.h"
#include "vine_pipe.h"      //init test
#include "core/vine_data.h" //free date
#include "core/vine_accel.h"//inc dec size
#include <pthread.h> 


#define GPU_SIZE    2097152
#define DATA_SIZE   997152
#define BIG_SIZE    1597152

async_meta_s meta;

const char config[] = "shm_file vt_test\n" "shm_size 0x10000000\n";

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
	proc = (vine_proc_s*)vine_proc_register(type,name,pd,psize);
	return proc;
}

START_TEST(test_gpu_size)
{
    vine_accel_s* accel;
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //init accel
    accel = vine_accel_init(mypipe, "FakeAccel", 1 , GPU_SIZE, GPU_SIZE*2);
    ck_assert(accel!=0);

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
    pthread_t * thread;
    size_t size_before;
    vine_accel_s *accel,*myaccel;
    vine_accel_type_e accelType = GPU; //GPU : 1
	
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //create proc
	vine_proc_s *process_id = create_proc(mypipe,accelType,"issue_proc",0,0);
    ck_assert( !!process_id );

    //initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE, GPU_SIZE*2);
    ck_assert(accel!=0);
    
    //acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
	ck_assert(!!myaccel);
    
    //set phys
    vine_accel_set_physical(myaccel,accel);
    
    //test inc
    size_before = vine_accel_get_avaliable_size(myaccel);
    thread = spawn_thread(size_inc,myaccel);
	wait_thread(thread);
    //check
    ck_assert_int_eq( vine_accel_get_avaliable_size(myaccel)        ,size_before+DATA_SIZE);
    
    //test dec
    size_before = vine_accel_get_avaliable_size(myaccel);
    thread = spawn_thread(size_dec,myaccel);
	wait_thread(thread);
    //check
    ck_assert_int_eq( vine_accel_get_avaliable_size(myaccel)        ,size_before-DATA_SIZE);
    
    
    //exit vine_talk
	vine_talk_exit();
}
END_TEST

START_TEST(test_thread_wait)
{
    //staff to use
    pthread_t * thread1,* thread2,*thread3,*thread4;
    size_t              size_before = 0;
    vine_accel_s        *accel,*myaccel;
    vine_accel_type_e   accelType = GPU; //GPU : 1
	
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //create proc
	vine_proc_s *process_id = create_proc(mypipe,accelType,"issue_proc",0,0);
    ck_assert( !!process_id );

    //initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE, GPU_SIZE*2);
    ck_assert(accel!=0);
    
    //acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
	ck_assert(!!myaccel);
    
    //set phys
    vine_accel_set_physical(myaccel,accel);
    
    //first dec
    size_before = vine_accel_get_avaliable_size(myaccel);
    thread1 = spawn_thread(size_big_dec,myaccel);
    wait_thread(thread1);
    ck_assert_int_eq(vine_accel_get_avaliable_size(myaccel),size_before - BIG_SIZE );
    
    //wait here
    size_before = vine_accel_get_avaliable_size(myaccel);
    thread1 = spawn_thread(size_big_dec,myaccel);
    thread3 = spawn_thread(size_big_dec,myaccel);
    usleep(1000);
    ck_assert_int_eq(vine_accel_get_avaliable_size(myaccel) ,size_before);
    
    thread2 = spawn_thread(size_big_inc,myaccel);
    usleep(1000);
    ck_assert_int_eq(vine_accel_get_avaliable_size(myaccel) ,size_before);
    
    thread4 = spawn_thread(size_big_inc,myaccel);
    usleep(1000);
    ck_assert_int_eq(vine_accel_get_avaliable_size(myaccel) ,size_before );
    
    wait_thread(thread4);
    wait_thread(thread3);
    wait_thread(thread2);
    wait_thread(thread1);
    ck_assert_int_eq(vine_accel_get_avaliable_size(myaccel) ,GPU_SIZE -  BIG_SIZE);
    
    //exit vine_talk
	vine_talk_exit();
}
END_TEST

struct arg_struct {
    vine_pipe_s         *mypipe     ;
    vine_task           *task       ; 
    vine_accel_type_e   accelType   ;
    vine_vaccel_s       *accelInUse ;
    vine_accel_s        *phys_accel ;
};

void *init_phys(void *data){
    struct arg_struct* args = (struct arg_struct*)data;
    vine_pipe_wait_for_task(args->mypipe,args->accelType);
	ck_assert_int_eq(vine_task_stat(args->task,0),task_issued);
    
    vine_task_msg_s* temp_task1 =(vine_task_msg_s*) utils_queue_pop(&args->accelInUse->queue);
    ck_assert_int_eq(vine_task_stat(temp_task1,0),task_issued);
    vine_task_msg_s* temp_task2 =(vine_task_msg_s*) utils_queue_pop(&args->accelInUse->queue);
    ck_assert_int_eq(vine_task_stat(temp_task2,0),task_issued);
    
    
    printf("Task isssue for something %s\n",((vine_proc_s*)((vine_task_msg_s*)(temp_task2))->proc)->obj.name);
    
    //set phys
    vine_accel_set_physical(args->accelInUse,args->phys_accel);
    
    vine_task_mark_done(temp_task2,task_completed);
    return NULL;
}


void *init_data_mark_done(void* data){
    // struct arg_struct* args = (struct arg_struct*)data;
    // //wait for task to come for exec.
	// vine_pipe_wait_for_task(args->mypipe,args->accelType);
	// ck_assert_int_eq(vine_task_stat(args->task,0),task_issued);
    
    // int boom_in[]={12,34,123,4,123,63,645,63,42};
    // int boom_out[]={12,34,123,4,2345,63,43565,63,42};
    // //init vine data
    // vine_buffer_s inputs[1] = {VINE_BUFFER(boom_in, DATA_SIZE)};
    // vine_buffer_s outputs[1] = {VINE_BUFFER(boom_out, DATA_SIZE)};
    // vine_assert(inputs != NULL);
    // vine_assert(outputs!= NULL);
    
    // //create proc
	// vine_proc_s *syncTo = create_proc(args->mypipe,args->accelType,"syncTo",0,0);
    // ck_assert( !!syncTo );
    // vine_proc_s *init_phy = create_proc(args->mypipe,args->accelType,"init_phys",0,0);
    // ck_assert( !!init_phy );
    

    // //issue task
    // vine_task *task = vine_task_issue(args->accelInUse, syncTo, 0, 0, 
    //                                     1, inputs, 1, outputs);
    
    // //check isssued
    // vine_pipe_wait_for_task(args->mypipe,args->accelType);
	// ck_assert_int_eq(vine_task_stat(args->task,0),task_issued);
    
    // //check data->remote and init
    // ck_assert_ptr_eq(((vine_data_s*)inputs[0])->remote,0);
    // ck_assert_ptr_eq(((vine_data_s*)inputs[0])->remote,0);
    // ((vine_data_s*)inputs[0])->remote = malloc(sizeof(DATA_SIZE));
    // ((vine_data_s*)outputs[0])->remote = malloc(sizeof(DATA_SIZE));
    // ck_assert( ((vine_data_s*)inputs[0] ) ->remote != NULL);
    // ck_assert( ((vine_data_s*)outputs[0]) ->remote != NULL);
    
    // //mark done after exec
	// vine_task_mark_done(task,task_completed);
	// ck_assert_int_eq(vine_task_stat(task,0),task_completed);
	// vine_task_wait_done(task);
    
    // //free task
    // ck_assert_int_eq(vine_object_refs((vine_object_s*)task),1);
	// vine_task_free(task);
    
    // //free data
    // vine_data_free(inputs[0]);
    // vine_data_free(outputs[0]);
    
    return 0;
}

START_TEST(test_single_phys_task_issue_without_wait)
{
    // //staff to use
    // struct arg_struct* data = malloc(sizeof(struct arg_struct));
    // pthread_t *thread1,*thread2;
    // vine_task_msg_s* temp_task;
	// vine_vaccel_s *myaccel;
	// vine_accel_s*accel;
    // vine_accel_type_e accelType = 1; //GPU
    // int i;
	
    // //init vine_talk
    // vine_pipe_s  *mypipe = vine_talk_init();
	// ck_assert(!!mypipe);
    
    // //create proc
	// vine_proc_s *process_id = create_proc(mypipe,accelType,"init_data",0,0);
    // ck_assert( !!process_id );
    
    // //create proc
	// vine_proc_s *free_id = create_proc(mypipe,accelType,"free",0,0);
    // ck_assert( !!free_id );

    // //initAccelerator
    // accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE, GPU_SIZE*2);
    // ck_assert(accel!=0);
    
    // //acquireAccelerator
    // myaccel = vine_accel_acquire_type(accelType);
	// ck_assert(!!myaccel);
     
    // //init vine_task to add size 
    // ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),1);
	// vine_task * task = vine_task_issue(myaccel,process_id,0,0,0,0,0,0);
	// ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),2);
    
    // //init argumments
    // data->task       = task;
    // data->mypipe     = mypipe;
    // data->accelType  = accelType;
    // data->accelInUse = myaccel;
    // data->phys_accel = accel;
    
    // //take task and init 
    // thread1 = spawn_thread(init_data_mark_done,(void*)data);
    // usleep(1000);
    // thread2 = spawn_thread(init_phys,(void*)data);
    // usleep(1000);
    // wait_thread(thread2);
    // wait_thread(thread1);
    // ck_assert_int_eq(accel->AvaliableSize,GPU_SIZE);
    // ck_assert_int_eq(accel->totalSize,GPU_SIZE*2);
    
    // //delete task
    // ck_assert_int_eq(vine_object_refs((vine_object_s*)task),1);
	// vine_task_free(task);
    
    // //clean task from pipe
    // for(i=0;i<utils_queue_used_slots(&myaccel->queue);i++){
    //     temp_task =(vine_task_msg_s*) utils_queue_pop(&myaccel->queue);
    //     //Check it
    //     printf("\t Task from pipe %s\n",((vine_proc_s*)((vine_task_msg_s*)(temp_task))->proc)->obj.name);
    //     //mark done after exec
    //     vine_task_mark_done(temp_task,task_completed);
    //     ck_assert_int_eq(vine_task_stat(temp_task,0),task_completed);
    //     vine_task_wait_done(temp_task);
        
    //     //free task
    //     if(vine_object_refs((vine_object_s*)temp_task) == 1){
    //         ck_assert_int_eq(vine_object_refs((vine_object_s*)temp_task),1);
    //         vine_task_free(temp_task);
    //         vine_object_ref_dec((vine_object_s*)myaccel);
    //     }else{
    //         vine_object_ref_dec((vine_object_s*)myaccel);
    //     }
        
    // }

    // //releaseAccelerator check ref_count check free done
	// ck_assert_int_eq(get_object_count(&(mypipe->objs),VINE_TYPE_VIRT_ACCEL),1);
    // ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),1);
	// vine_accel_release((vine_accel **)&myaccel);
    // ck_assert_ptr_eq(myaccel,0);
	
    // //exit vine_talk
	// vine_talk_exit();
}
END_TEST
/*
START_TEST(test_single_phys_task_issue_with_wait)
{
    struct arg_struct* data = malloc(sizeof(struct arg_struct));
    pthread_t *thread1,*thread2;//,thread3 ,*thread4
    vine_task_msg_s* temp_task;
    //staff to use
	vine_vaccel_s *myaccel;
	vine_accel_s*accel;
    vine_accel_type_e accelType = 1; //GPU
    int i;
	
    //init vine_talk
    vine_pipe_s  *mypipe = vine_talk_init();
	ck_assert(!!mypipe);
    
    //create proc
	vine_proc_s *process_id = create_proc(mypipe,accelType,"init_data",0,0);
    ck_assert( !!process_id );
    
    //create proc
	vine_proc_s *free_id = create_proc(mypipe,accelType,"free",0,0);
    ck_assert( !!free_id );

    //initAccelerator
    accel = vine_accel_init(mypipe, "FakeAccel", accelType, GPU_SIZE, GPU_SIZE);
    ck_assert(accel!=0);
    
    //acquireAccelerator
    myaccel = vine_accel_acquire_type(accelType);
	ck_assert(!!myaccel);
     
    //init vine_task to add size 
    ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),1);
	vine_task * task = vine_task_issue(myaccel,process_id,0,0,0,0,0,0);
	ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),2);
    
    //init argumments
    data->task       = task;
    data->mypipe     = mypipe;
    data->accelType  = accelType;
    data->accelInUse = myaccel;
    data->phys_accel = accel;
    
    //take task and init 
    thread1 = spawn_thread(init_data_mark_done,(void*)data);
    usleep(1000);
    thread2 = spawn_thread(init_phys,(void*)data);
    usleep(1000);
    wait_thread(thread2);
    wait_thread(thread1);
    ck_assert_int_eq(accel->AvaliableSize,GPU_SIZE);
    ck_assert_int_eq(accel->totalSize,GPU_SIZE);
    
    //delete task
    ck_assert_int_eq(vine_object_refs((vine_object_s*)task),1);
	vine_task_free(task);
    
    //clean task from pipe
    for(i=0;i<utils_queue_used_slots(&myaccel->queue);i++){
        temp_task =(vine_task_msg_s*) utils_queue_pop(&myaccel->queue);
        //Check it
        printf("\t Task from pipe %s\n",((vine_proc_s*)((vine_task_msg_s*)(temp_task))->proc)->obj.name);
        //mark done after exec
        vine_task_mark_done(temp_task,task_completed);
        ck_assert_int_eq(vine_task_stat(temp_task,0),task_completed);
        vine_task_wait_done(temp_task);
        
        //free task
        if(vine_object_refs((vine_object_s*)temp_task) == 1){
            ck_assert_int_eq(vine_object_refs((vine_object_s*)temp_task),1);
            vine_task_free(temp_task);
            vine_object_ref_dec((vine_object_s*)myaccel);
        }else{
            vine_object_ref_dec((vine_object_s*)myaccel);
        }
        
    }

    //releaseAccelerator check ref_count check free done
	ck_assert_int_eq(get_object_count(&(mypipe->objs),VINE_TYPE_VIRT_ACCEL),1);
    ck_assert_int_eq(vine_object_refs((vine_object_s*)myaccel),1);
	vine_accel_release((vine_accel **)&myaccel);
    ck_assert_ptr_eq(myaccel,0);
	
    //exit vine_talk
	vine_talk_exit();
}
END_TEST
*/

/*
START_TEST(test_assert_false)
{
    vine_assert(0);
}
END_TEST
*/

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
    tcase_add_test(tc_single, test_thread_wait);
    tcase_add_test(tc_single, test_single_phys_task_issue_without_wait);
    //tcase_add_test(tc_single, test_single_phys_task_issue_with_wait);
    //tcase_add_test_raise_signal(tc_single, test_assert_false,6);
    
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
