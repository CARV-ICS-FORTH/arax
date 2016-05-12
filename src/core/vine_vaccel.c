#include "vine_vaccel.h"

vine_vaccel_s* vine_vaccel_init(vine_object_repo_s *repo, void *mem,
                                size_t mem_size, char *name,
                                vine_accel_s *accel)
{
	vine_vaccel_s *vaccel = mem;

	utils_spinlock_init( &(vaccel->lock) );
	if ( !utils_queue_init( vaccel+1, mem_size-sizeof(*vaccel) ) )
		;
	return 0;
	vine_object_register(repo, &(vaccel->obj), VINE_TYPE_VIRT_ACCEL, name);
	return vaccel;
}

utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel)
{
	return (utils_queue_s*)(vaccel+1);
}

void vine_vaccel_erase(vine_object_repo_s *repo, vine_vaccel_s *vaccel)
{
	vine_object_remove( repo, &(vaccel->obj) );
}
