#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <check.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <conf.h>
#include "vine_pipe.h"
#include "utils/system.h"
#include "utils/config.h"

static int __attribute__( (unused) ) test_file_exists(char *file)
{
	struct stat buf;

	return !stat(file, &buf);
}

static int test_rename(const char * src,const char * dst)
{
	int ret = rename(src,dst);

	if( ret )
		printf("Renaming %s to %s failed:%s\n",src,dst,strerror(ret));

	return ret;
}

/**
 * Backup current config VINE_CONFIG_FILE to ./vinetalk.bak.
 */
static void __attribute__( (unused) ) test_backup_config()
{
	char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

	if ( test_file_exists(conf_file) )
		ck_assert( !test_rename(conf_file, "vinetalk.bak") ); /* Keep old
	                                                          * file */

	ck_assert_int_eq(system_file_size(conf_file),0);

	utils_config_free_path(conf_file);
}

/**
 * Restore ./vinetalk.bak. to VINE_CONFIG_FILE.
 */
static void __attribute__( (unused) ) test_restore_config()
{
	char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

	ck_assert( !unlink(conf_file) ); /* Remove test file*/
	if ( test_file_exists("vinetalk.bak") )
		ck_assert( !test_rename("vinetalk.bak", conf_file) );

	utils_config_free_path(conf_file);
}

/**
 * Open config file at VINE_CONFIG_FILE.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
static int __attribute__( (unused) ) test_open_config()
{
	int  fd;
	char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

	fd = open(conf_file, O_RDWR|O_CREAT, 0666);
	ck_assert_int_gt(fd, 0);
	ck_assert_int_eq(system_file_size(conf_file),lseek(fd,0,SEEK_END));
	utils_config_free_path(conf_file);
	return fd;
}

static __attribute__( (unused) ) pthread_t * spawn_thread(void * (func)(void*),void * data)
{
	pthread_t * thread = malloc(sizeof(*thread));
	ck_assert(!!thread);
	pthread_create(thread,0,func,data);
	return thread;
}

static __attribute__( (unused) ) void wait_thread(pthread_t * thread)
{
	ck_assert(!!thread);
	pthread_join(*thread,0);
	free(thread);
}

static __attribute__( (unused) ) int get_object_count(vine_object_repo_s  *repo,vine_object_type_e type)
{
	int ret =
	vine_object_list_lock(repo,type)->length;
	vine_object_list_unlock(repo,type);
	return ret;
}

typedef struct {
	int tasks;
	vine_accel_type_e type;
} n_task_handler_state;

#define SEC_IN_USEC (1000*1000)

static __attribute__( (unused) ) void safe_usleep(int64_t us)
{
	struct timespec rqtp;
	
	if(us >= SEC_IN_USEC)	// Time is >= 1s
		rqtp.tv_sec = us / SEC_IN_USEC;
	else
		rqtp.tv_sec = 0;
	
	us -= rqtp.tv_sec*SEC_IN_USEC;	// Remote the 'full' seconds
	
	rqtp.tv_nsec = us*1000;
	
	while( nanosleep(&rqtp,&rqtp) != 0 );
}

void * n_task_handler(void * data)
{
	n_task_handler_state * state = data;

	vine_pipe_s * vpipe = vine_talk_init();

	ck_assert_int_eq(get_object_count(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL),1);

	safe_usleep(100000);

	while(state->tasks)
	{
		printf("%s(%d)\n",__func__,state->tasks);
		vine_pipe_wait_for_task_type_or_any_assignee(vpipe,state->type,0);
		utils_list_s * vacs = vine_object_list_lock(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL);
		utils_list_node_s * vac_node;
		vine_vaccel_s * vac;
		vine_task_msg_s * task;
		utils_list_for_each(*vacs,vac_node)
		{
			vac = vac_node->owner;
            vine_accel_set_physical(vac, (void*)0xF00DF00D);
			ck_assert_int_eq(vac->obj.type,VINE_TYPE_VIRT_ACCEL);
			task = utils_queue_pop(vine_vaccel_queue(vac));
			ck_assert(task);
			vine_proc_s * proc = task->proc;
			printf("Executing a '%s' task!\n",proc->obj.name);
			vine_task_mark_done(task,task_completed);
		}
		vine_object_list_unlock(&(vpipe->objs),VINE_TYPE_VIRT_ACCEL);
		state->tasks--;
	}
	printf("%s(%d)\n",__func__,state->tasks);
	free(state);
	return 0;
}

static __attribute__( (unused) ) void handle_n_tasks(int tasks,vine_accel_type_e type)
{
	n_task_handler_state * state = malloc(sizeof(*state));
	state->tasks = tasks;
	state->type = type;
	spawn_thread(n_task_handler,(void*)state);
}


#endif /* ifndef TESTING_HEADER */
