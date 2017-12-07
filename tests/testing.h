#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <check.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
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

/**
 * Backup current config VINE_CONFIG_FILE to ./vinetalk.bak.
 */
static void __attribute__( (unused) ) test_backup_config()
{
	char *conf_file = utils_config_alloc_path(VINE_CONFIG_FILE);

	if ( test_file_exists(conf_file) )
		ck_assert( !rename(conf_file, "vinetalk.bak") ); /* Keep old
	                                                          * file */
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
		ck_assert( !rename("vinetalk.bak", conf_file) );

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

#endif /* ifndef TESTING_HEADER */
