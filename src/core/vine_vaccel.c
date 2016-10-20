#include "vine_vaccel.h"

vine_vaccel_s* vine_vaccel_init(vine_object_repo_s *repo, void *mem,
                                size_t mem_size, const char *name,
								vine_accel_type_e  type,vine_accel_s *accel)
{
	vine_vaccel_s *vaccel = mem;
	vaccel->phys = accel;
	utils_spinlock_init( &(vaccel->lock) );
	if ( !utils_queue_init( vaccel+1 ) )
		return 0;
	utils_list_node_init(&(vaccel->vaccels),vaccel);
	vaccel->type = type;
	vine_object_register(repo, &(vaccel->obj), VINE_TYPE_VIRT_ACCEL, name);
	if(accel)
		vine_accel_add_vaccel(accel,vaccel);
	return vaccel;
}

utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel)
{
	if(vaccel->obj.type != VINE_TYPE_VIRT_ACCEL)
		return 0;	/* That was not a vine_vaccel_s */
	return (utils_queue_s*)(vaccel+1);
}

int vine_vaccel_erase(vine_object_repo_s *repo, vine_vaccel_s *vaccel)
{
	if(vaccel->obj.type != VINE_TYPE_VIRT_ACCEL)
		return 0;
	if(vaccel->phys)
		vine_accel_del_vaccel(vaccel->phys,vaccel);
	vaccel->type = VINE_ACCEL_TYPES;	// Should be freed by the controller
	vine_object_remove( repo, &(vaccel->obj) );
	return 1;
}

int vine_vaccel_reclaim(arch_alloc_s *alloc,vine_vaccel_s *vaccel)
{
	if(vaccel->type != VINE_ACCEL_TYPES)
		return 0;
	arch_alloc_free(alloc,vaccel);
	return 1;
}

vine_accel_state_e vine_vaccel_get_stat(vine_vaccel_s *accel,vine_accel_stats_s * stat)
{
	return vine_accel_get_stat(accel->phys,stat);
}
