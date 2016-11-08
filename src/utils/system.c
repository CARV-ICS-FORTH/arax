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

int system_compare_ptrs(const void * a,const void * b)
{
	return (int)((size_t)a - (size_t)b);
}

off_t system_file_size(const char * file)
{
	struct stat stats = {0};
	if(stat(file,&stats))
		return 0;
	return stats.st_size;
}
