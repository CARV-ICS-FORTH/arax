#include "vine_accel.h"
#include <string.h>

vine_accel_s* vine_accel_init(vine_object_repo_s *repo,async_meta_s * meta, void *mem, const char *name,
                              vine_accel_type_e type)
{
	vine_accel_s *obj = mem;
	vine_object_register(repo, &(obj->obj), VINE_TYPE_PHYS_ACCEL, name);
	utils_spinlock_init(&(obj->lock));
	utils_list_init(&(obj->vaccels));
	obj->type = type;
	obj->state = accel_idle;
	obj->revision = 0;
#ifdef QRS_ENABLE
	async_completion_init(meta, &(obj->tasks_to_run));
#endif
	return obj;
}

size_t vine_accel_calc_size(const char *name)
{
	return sizeof(vine_accel_s);
}

const char* vine_accel_get_name(vine_accel_s *accel)
{
	return accel->obj.name;
}

vine_accel_state_e vine_accel_get_stat(vine_accel_s *accel,vine_accel_stats_s * stat)
{
	/* TODO: IMPLEMENT stat memcpy */
	return accel->state;
}

void vine_accel_inc_revision(vine_accel_s * accel)
{
	__sync_fetch_and_add(&(accel->revision),1);
}

size_t vine_accel_get_revision(vine_accel_s * accel)
{
	return accel->revision;
}

void vine_accel_wait_for_tasks(async_meta_s * meta,vine_accel_s * accel)
{
#ifdef QRS_ENABLE
	async_completion_wait(meta,&(accel->tasks_to_run));
#endif
}

void vine_accel_add_task(async_meta_s * meta,vine_accel_s * accel)
{
#ifdef QRS_ENABLE
	async_completion_complete(meta,&(accel->tasks_to_run));
#endif
}

void vine_accel_add_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel)
{
	utils_spinlock_lock(&(accel->lock));
	utils_list_add(&(accel->vaccels),&(vaccel->vaccels));
	utils_spinlock_unlock(&(accel->lock));
	vine_accel_inc_revision(accel);
}

void vine_accel_del_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel)
{
	utils_spinlock_lock(&(accel->lock));
	utils_list_del(&(accel->vaccels),&(vaccel->vaccels));
	utils_spinlock_unlock(&(accel->lock));
	vine_accel_inc_revision(accel);
}

void vine_accel_erase(vine_object_repo_s *repo,vine_accel_s * accel)
{
	utils_spinlock_lock(&(accel->lock));
	if(accel->vaccels.length)
		fprintf(stderr,"Erasing physical accelerator %s "
		"with %lu attached virtual accelerators!\n",
		accel->obj.name,accel->vaccels.length);
	utils_spinlock_unlock(&(accel->lock));
	vine_object_remove(repo,&(accel->obj));
}
