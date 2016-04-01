#include "vine_accel.h"
#include <string.h>

vine_accel_s* vine_accel_init(void *mem, char *name, vine_accel_type_e type)
{
	vine_accel_s *obj = mem;

	structs_list_node_init( &(obj->list) );
	obj->owner = 0;
	obj->type  = type;
}

size_t vine_accel_calc_size(char *name)
{
	return sizeof(vine_accel_s)+strlen(name);
}
