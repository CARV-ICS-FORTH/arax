#include "vine_accel.h"
#include <string.h>

vine_accel_s* vine_accel_init(void *mem, char *name, vine_accel_type_e type)
{
	vine_accel_s *obj = mem;

	utils_list_node_init( &(obj->list) );
	obj->owner = 0;
	obj->type  = type;
	memcpy(obj->name,name,strlen(name));
	return obj;
}

size_t vine_accel_calc_size(char *name)
{
	return sizeof(vine_accel_s)+strlen(name);
}

const char * vine_accel_get_name(vine_accel_s* accel)
{
	return accel->name;
}
