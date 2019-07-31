#include "vine_proc.h"
#include <string.h>

vine_proc_s* vine_proc_init(vine_object_repo_s *repo, const char *name,
							vine_accel_type_e type, const void *code,
							size_t code_size)
{
	vine_proc_s *proc =
	(vine_proc_s*)vine_object_register(repo, VINE_TYPE_PROC, name,
									   sizeof(vine_proc_s)+code_size,1);

	if(!proc)
		return 0;

	proc->type     = type;
	proc->users    = 0;
	proc->bin_size = code_size;
	utils_breakdown_init_stats(&(proc->breakdown));
	memcpy(proc+1, code, code_size);
	return proc;
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

VineFunctor * vine_proc_get_functor(vine_proc_s *proc)
{
	return *((VineFunctor **)vine_proc_get_code(proc,0));
}

int vine_proc_mod_users(vine_proc_s *proc, int delta)
{
	return __sync_fetch_and_add(&(proc->users), delta);
}

VINE_OBJ_DTOR_DECL(vine_proc_s)
{
}
