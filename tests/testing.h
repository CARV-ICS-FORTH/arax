#ifndef TESTING_HEADER
#define TESTING_HEADER
#include <check.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include "utils/system.h"

static int __attribute__ ((unused)) test_file_exists(char * file)
{
	struct stat buf;
	return !stat ( file, &buf );

}

/**
 * Backup current config ~/.vinetalk to ./vinetalk.bak.
 */
static void __attribute__ ((unused)) test_backup_config()
{
	char vtpath[1024];
	printf("%s\n",__func__);
	snprintf(vtpath,1024,"%s/.vinetalk",system_home_path());
	if(test_file_exists(vtpath))
		ck_assert(!rename(vtpath,"vinetalk.bak")); /* Keep old file */
}

/**
 * Restore ./vinetalk.bak. to ~/.vinetalk.
 */
static void __attribute__ ((unused)) test_restore_config()
{
	char vtpath[1024];
	printf("%s\n",__func__);
	snprintf(vtpath,1024,"%s/.vinetalk",system_home_path());
	ck_assert(!unlink(vtpath));					/* Remove test file*/
	if(test_file_exists("vinetalk.bak"))
		ck_assert(!rename("vinetalk.bak",vtpath));
}

/**
 * Open config file at ~/.vinetalk.
 * \note use close() to close returned file descriptor.
 * @return File descriptor of the configuration file.
 */
static int __attribute__ ((unused)) test_open_config()
{
	int fd;
	char vtpath[1024];
	printf("%s\n",__func__);
	snprintf(vtpath,1024,"%s/.vinetalk",system_home_path());
	fd = open(vtpath,O_RDWR|O_CREAT,0666);
	ck_assert_int_gt(fd,0);
	return fd;
}
#endif /* ifndef TESTING_HEADER */
