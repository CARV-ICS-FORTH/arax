#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <check.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "utils/system.h"

/**
 * Backup current config ~/.vinetalk to ./vinetalk.bak.
 */
static void test_backup_config()
{
	char vtpath[1024];
	snprintf(vtpath,1024,"%s/.vinetalk",system_home_path());
	rename(vtpath,"vinetalk.bak"); /* Keep old file */
}

/**
 * Restore ./vinetalk.bak. to ~/.vinetalk.
 */
static void test_restore_config()
{
	char vtpath[1024];
	snprintf(vtpath,1024,"%s/.vinetalk",system_home_path());
	unlink(vtpath);					/* Remove test file*/
	rename("vinetalk.bak",vtpath);
}

/**
 * Open config file at ~/.vinetalk.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
static int test_open_config()
{
	int fd;
	char vtpath[1024];
	snprintf(vtpath,1024,"%s/.vinetalk",system_home_path());
	fd = open(vtpath,O_RDWR|O_CREAT,0666);
	ck_assert_int_gt(fd,0);
	return fd;
}
#endif /* ifndef TESTING_HEADER */
