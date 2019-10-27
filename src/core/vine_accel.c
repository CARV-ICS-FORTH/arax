#include "vine_accel.h"
#include "arch/alloc.h"
#include "vine_pipe.h"
#include "utils/vine_assert.h"
#include <string.h>

vine_accel_s* vine_accel_init(vine_pipe_s * pipe, const char *name,
                              vine_accel_type_e type,size_t size)
{
	vine_accel_s *obj = (vine_accel_s *)vine_object_register(&(pipe->objs),
											 VINE_TYPE_PHYS_ACCEL,
										  name, sizeof(vine_accel_s),1);

	if(!obj)				// GCOV_EXCL_LINE
		return obj;			// GCOV_EXCL_LINE
	utils_spinlock_init(&(obj->lock));
	utils_list_init(&(obj->vaccels));
	obj->type = type;
	obj->state = accel_idle;
	obj->revision = 0;
    obj->AvaliableSize = size;
#ifdef QRS_ENABLE
	async_completion_init(meta, &(obj->tasks_to_run));
#endif
	return obj;
}

void vine_accel_size_inc(vine_accel* vaccel,size_t sz){
	vine_assert(vaccel);
	vine_vaccel_s*    acl    = (vine_vaccel_s*)vaccel;
	vine_assert(acl);
	vine_accel_s*  	  phys 	 = acl->phys;
	vine_assert(phys);
	__sync_fetch_and_add( &(phys->AvaliableSize) , sz );
}

void vine_accel_size_dec(vine_accel* vaccel,size_t sz){
	vine_assert(vaccel);
	vine_vaccel_s*    acl    = (vine_vaccel_s*)vaccel;
	vine_assert(acl);
	vine_accel_s*  	  phys 	 = (vine_accel_s*)acl->phys;
	vine_assert(phys);
	__sync_fetch_and_sub(&(phys->AvaliableSize) , sz);
}

size_t vine_accel_get_size(vine_accel* vaccel){
	vine_vaccel_s*    acl    = (vine_vaccel_s*)vaccel;
	vine_assert(acl);
	vine_accel_s*  	  phys 	 = (vine_accel_s*)acl->phys;
	vine_assert(phys);
	return phys->AvaliableSize;
}

size_t vine_accel_get_AvaliableSize(vine_accel_s* accel){
	return accel->AvaliableSize;
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

void vine_accel_add_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel)
{
	vine_object_ref_inc(&(vaccel->obj));
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
	vine_object_ref_dec(&(vaccel->obj));
	vine_accel_inc_revision(accel);
}

VINE_OBJ_DTOR_DECL(vine_accel_s)
{
	vine_accel_s * accel = (vine_accel_s *)obj;
	utils_spinlock_lock(&(accel->lock));
	if(accel->vaccels.length)
		fprintf(stderr,"Erasing physical accelerator %s "
		"with %lu attached virtual accelerators!\n",
		accel->obj.name,accel->vaccels.length);
	utils_spinlock_unlock(&(accel->lock));

	arch_alloc_free(obj->repo->alloc,obj);
}
