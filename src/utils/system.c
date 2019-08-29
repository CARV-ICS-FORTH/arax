#include "system.h"
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

size_t system_total_memory()
{
	size_t pages     = sysconf(_SC_PHYS_PAGES);
	size_t page_size = sysconf(_SC_PAGE_SIZE);

	return pages * page_size;
}

char* system_home_path()
{
	uid_t         uid = getuid();
	struct passwd *pw = getpwuid(uid);

	if (!pw)
		return 0;

	return pw->pw_dir;
}

off_t system_file_size(const char * file)
{
	struct stat stats = {0};
	if(stat(file,&stats))
		return 0;
	return stats.st_size;
}

const char * system_exec_name()
{
	static char exec_name[1024];
	exec_name[readlink("/proc/self/exe", exec_name, 1023)]=0;
	return exec_name;
}

int system_process_id()
{
	return getpid();
}
