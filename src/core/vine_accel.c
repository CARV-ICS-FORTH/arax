#include "vine_accel.h"
#include <string.h>

vine_accel_s* vine_accel_init(vine_object_repo_s * repo,void *mem, char *name, vine_accel_type_e type)
{
	vine_accel_s *obj = mem;

	vine_object_register(repo,&(obj->obj),VINE_TYPE_PHYS_ACCEL,name);
	obj->owner = 0;
	obj->type  = type;
	return obj;
}

size_t vine_accel_calc_size(char *name)
{
	return sizeof(vine_accel_s);
}

const char* vine_accel_get_name(vine_accel_s *accel)
{
	return accel->obj.name;
}
