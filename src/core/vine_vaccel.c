#include "vine_vaccel.h"
#include "vine_pipe.h"
#include "arch/alloc.h"

vine_vaccel_s* vine_vaccel_init(vine_pipe_s * pipe, const char *name,
								vine_accel_type_e  type,vine_accel_s *accel)
{
	vine_vaccel_s *vaccel = (vine_vaccel_s *)
	vine_object_register(&(pipe->objs), VINE_TYPE_VIRT_ACCEL, name, sizeof(vine_vaccel_s),1);

	if(!vaccel)		// GCOV_EXCL_LINE
		return 0;	// GCOV_EXCL_LINE

	async_condition_init(&(pipe->async),&(vaccel->cond_done));
	vaccel->task_done = 0;

	vaccel->phys = accel;
	vaccel->cid = (uint64_t)-1;
	vaccel->priority = (uint64_t)-1;
	utils_spinlock_init( &(vaccel->lock) );
	if ( !utils_queue_init( &(vaccel->queue) ) )	// GCOV_EXCL_LINE
		return 0;									// GCOV_EXCL_LINE
	utils_list_node_init(&(vaccel->vaccels),vaccel);
	vaccel->type = type;
	vaccel->meta = 0;
	vaccel->assignee = 0;
	vaccel->ordering = SEQUENTIAL;
	if(accel)
		vine_accel_add_vaccel(accel,vaccel);
	return vaccel;
}

void * vine_vaccel_test_set_assignee(vine_accel_s *accel,void * assignee)
{
    vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vine_vaccel_s *vaccel = (vine_vaccel_s *)accel;

	if(vaccel->ordering == SEQUENTIAL)
	{
		if(__sync_bool_compare_and_swap(&(vaccel->assignee),0,assignee))
			return assignee;	// Was unassigned
		if(__sync_bool_compare_and_swap(&(vaccel->assignee),assignee,assignee))
			return assignee;	// Was assigned to me
		return 0;
	}
	else
	{
		vaccel->assignee = assignee;
		return assignee;
	}
}

void * vine_vaccel_get_assignee(vine_accel_s *accel)
{
	vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vine_vaccel_s *vaccel = (vine_vaccel_s *)accel;
	return vaccel->assignee;
}

void vine_vaccel_set_ordering(vine_accel_s *accel, vine_accel_ordering_e ordering)
{
	vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vine_vaccel_s *vaccel = (vine_vaccel_s *)accel;
	vaccel->ordering = ordering;
}

vine_accel_ordering_e vine_vaccel_get_ordering(vine_accel_s *accel)
{
	vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vine_vaccel_s *vaccel = (vine_vaccel_s *)accel;
	return vaccel->ordering;
}


uint64_t vine_vaccel_set_cid(vine_vaccel_s *vaccel,uint64_t cid)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vaccel->cid = cid;
	return vaccel->cid;
}

uint64_t vine_vaccel_get_cid(vine_vaccel_s *vaccel)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	return vaccel->cid;
}

uint64_t vine_vaccel_set_job_priority(vine_vaccel_s *vaccel,uint64_t priority)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vaccel->priority = priority;
	return vaccel->priority;
}

uint64_t vine_vaccel_get_job_priority(vine_vaccel_s *vaccel)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	return vaccel->priority;
}


void vine_vaccel_set_meta(vine_vaccel_s *vaccel,void * meta)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	vaccel->meta = meta;
}

void * vine_vaccel_get_meta(vine_vaccel_s *vaccel)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	return vaccel->meta;
}

utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	return &(vaccel->queue);
}

unsigned int vine_vaccel_queue_size(vine_vaccel_s *vaccel)
{
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);
	return utils_queue_used_slots(vine_vaccel_queue(vaccel));
}

vine_accel_state_e vine_vaccel_get_stat(vine_vaccel_s *accel,vine_accel_stats_s * stat)
{
	vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	return vine_accel_get_stat(accel->phys,stat);
}

void vine_vaccel_wait_task_done(vine_vaccel_s *accel)
{
	vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	async_condition_lock(&(accel->cond_done));
	while(!accel->task_done)
		async_condition_wait(&(accel->cond_done));
	accel->task_done--;
	async_condition_unlock(&(accel->cond_done));
}

void vine_vaccel_mark_task_done(vine_vaccel_s *accel)
{
	vine_assert(accel->obj.type == VINE_TYPE_VIRT_ACCEL);
	async_condition_lock(&(accel->cond_done));
	accel->task_done++;
	async_condition_notify(&(accel->cond_done));
	async_condition_unlock(&(accel->cond_done));
}

VINE_OBJ_DTOR_DECL(vine_vaccel_s)
{
	vine_vaccel_s * vaccel = (vine_vaccel_s *)obj;
	vine_assert(vaccel->obj.type == VINE_TYPE_VIRT_ACCEL);

	if(vaccel->phys)
		vine_accel_del_vaccel(vaccel->phys,vaccel);

	arch_alloc_free(obj->repo->alloc,obj);
}
