#include "vine_proc.h"
#include <string.h>

vine_proc_s* vine_proc_init(void *mem, const char *name, vine_accel_type_e type,
                            const void *code, size_t code_size)
{
	vine_proc_s *proc = (vine_proc_s*)mem;

	vine_list_node_init( &(proc->list) );
	proc->type     = type;
	proc->users    = 0;
	proc->data_off = strlen(name);
	sprintf(proc->name, "%s", name);
	memcpy(proc->name+proc->data_off, code, code_size);
}

size_t vine_proc_calc_size(const char *name, size_t code_size)
{
	return sizeof(vine_proc_s)+strlen(name)+1+code_size;
}
