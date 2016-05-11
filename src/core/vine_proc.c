#include "vine_proc.h"
#include <string.h>

vine_proc_s* vine_proc_init(vine_object_repo_s * repo,void *mem, const char *name, vine_accel_type_e type,
                            const void *code, size_t code_size)
{
	vine_proc_s *proc = (vine_proc_s*)mem;
	vine_object_register(repo,&(proc->obj),VINE_TYPE_PROC,name);
	proc->type     = type;
	proc->users    = 0;
	proc->bin_size = code_size;
	memcpy(proc+1, code, code_size);
	return proc;
}

size_t vine_proc_calc_size(const char *name, size_t code_size)
{
	return sizeof(vine_proc_s)+code_size;
}

int vine_proc_match_code(vine_proc_s *proc, const void *code, size_t code_size)
{
	if (code_size != proc->bin_size)
		return 0;
	return !memcmp(code, proc+1, code_size);
}

void* vine_proc_get_code(vine_proc_s *proc, size_t *code_size)
{
	if (code_size)
		*code_size = proc->bin_size;
	return proc+1;
}

int vine_proc_mod_users(vine_proc_s *proc, int delta)
{
	return __sync_fetch_and_add(&(proc->users), delta);
}
