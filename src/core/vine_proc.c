#include "vine_proc.h"
#include <string.h>

vine_proc_s* vine_proc_init(void *mem, const char *name, vine_accel_type_e type,
                            const void *code, size_t code_size)
{
	vine_proc_s *proc = (vine_proc_s*)mem;

	utils_list_node_init( &(proc->list) );
	proc->type     = type;
	proc->users    = 0;
	proc->data_off = strlen(name);
	proc->bin_size = code_size;
	sprintf(proc->name, "%s", name);
	memcpy(proc->name+proc->data_off, code, code_size);
	return proc;
}

size_t vine_proc_calc_size(const char *name, size_t code_size)
{
	return sizeof(vine_proc_s)+strlen(name)+1+code_size;
}

int vine_proc_code_match(vine_proc_s* proc,const void * code,size_t code_size)
{
	if(code_size != proc->bin_size)
		return 0;
	return !memcmp(code,proc->name+proc->data_off,code_size);
}

int vine_proc_mod_users(vine_proc_s* proc,int delta)
{
	return __sync_fetch_and_add(&(proc->users),delta);
}
