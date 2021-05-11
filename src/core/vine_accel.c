#include "vine_pipe.h"
#include <string.h>
#include <stdlib.h>

vine_accel_s* vine_accel_init(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type, size_t size, size_t capacity)
{
    vine_accel_s *obj = (vine_accel_s *) vine_object_register(&(pipe->objs),
        VINE_TYPE_PHYS_ACCEL,
        name, sizeof(vine_accel_s), 1);

    if (!obj)        // GCOV_EXCL_LINE
        return obj;  // GCOV_EXCL_LINE

    utils_spinlock_init(&(obj->lock));
    async_semaphore_init(&(pipe->async), &(obj->tasks));
    utils_list_init(&(obj->vaccels));
    obj->type     = type;
    obj->state    = accel_idle;
    obj->revision = 0;
    vine_throttle_init(&(pipe->async), &(obj->throttle), size, capacity);
    return obj;
}

void vine_accel_wait_for_task(vine_accel_s *accel)
{
    async_semaphore_dec(&(accel->tasks));
}

void vine_accel_add_task(vine_accel_s *accel)
{
    async_semaphore_inc(&(accel->tasks));
}

int vine_accel_pending_tasks(vine_accel_s *accel)
{
    return async_semaphore_value(&(accel->tasks));
}

void VINE_THROTTLE_DEBUG_ACCEL_FUNC(vine_accel_size_inc)(vine_accel * accel,
  size_t sz VINE_THROTTLE_DEBUG_ACCEL_PARAMS){
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_accel_s *phys = accel;

    vine_throttle_size_inc(&(phys->throttle), sz);
}

void VINE_THROTTLE_DEBUG_ACCEL_FUNC(vine_accel_size_dec)(vine_accel * accel,
  size_t sz VINE_THROTTLE_DEBUG_ACCEL_PARAMS){
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_accel_s *phys = accel;

    vine_throttle_size_dec(&(phys->throttle), sz);
}

size_t vine_accel_get_available_size(vine_accel *accel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_accel_s *phys = accel;

    return vine_throttle_get_available_size(&(phys->throttle));
}

size_t vine_accel_get_total_size(vine_accel *accel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_accel_s *phys = accel;

    return vine_throttle_get_total_size(&(phys->throttle));
}

const char* vine_accel_get_name(vine_accel_s *accel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    return accel->obj.name;
}

vine_accel_state_e vine_accel_get_stat(vine_accel_s *accel, vine_accel_stats_s *stat)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    /* TODO: IMPLEMENT stat memcpy */
    return accel->state;
}

void vine_accel_inc_revision(vine_accel_s *accel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    __sync_fetch_and_add(&(accel->revision), 1);
}

size_t vine_accel_get_revision(vine_accel_s *accel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    return accel->revision;
}

void vine_accel_add_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);

    if ( (vaccel->phys) == accel)
        return;

    vine_assert(vaccel->phys == 0);
    utils_spinlock_lock(&(accel->lock));
    utils_list_add(&(accel->vaccels), &(vaccel->vaccels));
    utils_spinlock_unlock(&(accel->lock));

    int tasks = vine_vaccel_queue_size(vaccel);

    while (tasks--)
        vine_accel_add_task(accel);

    vine_accel_inc_revision(accel);
    vaccel->phys = accel;
}

size_t vine_accel_get_assigned_vaccels(vine_accel_s *accel, vine_vaccel_s **vaccel)
{
    size_t count = 0;

    utils_spinlock_lock(&(accel->lock));
    count   = accel->vaccels.length;
    *vaccel = malloc(sizeof(vine_vaccel_s *) * count);
    utils_list_to_array(&(accel->vaccels), (void **) vaccel);
    utils_spinlock_unlock(&(accel->lock));

    return count;
}

void vine_accel_del_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    vine_assert_obj(vaccel->phys, VINE_TYPE_PHYS_ACCEL);

    utils_spinlock_lock(&(accel->lock));
    utils_list_del(&(accel->vaccels), &(vaccel->vaccels));
    utils_spinlock_unlock(&(accel->lock));
    vine_accel_inc_revision(accel);
    vaccel->phys = 0;
}

VINE_OBJ_DTOR_DECL(vine_accel_s)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_accel_s *accel = (vine_accel_s *) obj;

    utils_spinlock_lock(&(accel->lock));
    if (accel->vaccels.length) {
        fprintf(stderr, "Erasing physical accelerator %s "
          "with %lu attached virtual accelerators!\n",
          accel->obj.name, accel->vaccels.length);
        vine_assert("Erasing physical accelerator with dangling virtual accels");
    }
    utils_spinlock_unlock(&(accel->lock));

    arch_alloc_free(obj->repo->alloc, obj);
}
