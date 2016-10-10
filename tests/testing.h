#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <check.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include "utils/system.h"
#include "utils/config.h"

static int __attribute__( (unused) ) test_file_exists(char *file)
{
	struct stat buf;

	return !stat(file, &buf);
}

/**
 * Return full path to .vinetalk file.
 */
static char* test_get_config_file()
{
	return utils_config_alloc_path("~/.vinetalk");
}

/**
 * Backup current config ~/.vinetalk to ./vinetalk.bak.
 */
static void __attribute__( (unused) ) test_backup_config()
{
	char *conf_file = test_get_config_file();

	if ( test_file_exists(conf_file) )
		ck_assert( !rename(conf_file, "vinetalk.bak") ); /* Keep old
	                                                          * file */
}

/**
 * Restore ./vinetalk.bak. to ~/.vinetalk.
 */
static void __attribute__( (unused) ) test_restore_config()
{
	char *conf_file = test_get_config_file();

	ck_assert( !unlink(conf_file) ); /* Remove test file*/
	if ( test_file_exists("vinetalk.bak") )
		ck_assert( !rename("vinetalk.bak", conf_file) );
}

/**
 * Open config file at ~/.vinetalk.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
static int __attribute__( (unused) ) test_open_config()
{
	int  fd;
	char *conf_file = test_get_config_file();

	fd = open(conf_file, O_RDWR|O_CREAT, 0666);
	ck_assert_int_gt(fd, 0);
	return fd;
}

#endif /* ifndef TESTING_HEADER */
