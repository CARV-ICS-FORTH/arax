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

    async_condition_init(&(pipe->async), &(obj->lock));
    obj->tasks = 0;
    utils_list_init(&(obj->vaccels));
    obj->type     = type;
    obj->state    = accel_idle;
    obj->revision = 0;
    vine_throttle_init(&(pipe->async), &(obj->throttle), size, capacity);
    obj->free_vaq = vine_vaccel_init(pipe, name, type, obj);
    return obj;
}

void vine_accel_wait_for_task(vine_accel_s *accel)
{
    async_condition_lock(&(accel->lock));

    while (accel->tasks == 0)
        async_condition_wait(&(accel->lock));

    accel->tasks--;

    async_condition_unlock(&(accel->lock));
}

void vine_accel_add_task(vine_accel_s *accel)
{
    async_condition_lock(&(accel->lock));
    accel->tasks++;
    async_condition_notify(&(accel->lock));
    async_condition_unlock(&(accel->lock));
}

size_t vine_accel_pending_tasks(vine_accel_s *accel)
{
    return accel->tasks;
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

    vine_pipe_remove_orphan_vaccel(vaccel->obj.repo->pipe, vaccel);

    utils_spinlock_lock(&(vaccel->lock));

    if ( (vaccel->phys) != accel) {
        async_condition_lock(&(accel->lock));

        utils_list_add(&(accel->vaccels), &(vaccel->vaccels));

        int tasks = vine_vaccel_queue_size(vaccel);

        if (tasks) {
            accel->tasks += tasks;
            async_condition_notify(&(accel->lock));
        }

        vaccel->phys = accel;

        async_condition_unlock(&(accel->lock));
        vine_accel_inc_revision(accel);
    }

    utils_spinlock_unlock(&(vaccel->lock));
} /* vine_accel_add_vaccel */

size_t vine_accel_get_assigned_vaccels(vine_accel_s *accel, vine_vaccel_s ***vaccel)
{
    size_t count = 0;

    async_condition_lock(&(accel->lock));
    count   = accel->vaccels.length;
    *vaccel = malloc(sizeof(vine_vaccel_s *) * count);
    utils_list_to_array(&(accel->vaccels), (void **) *vaccel);
    async_condition_unlock(&(accel->lock));

    return count;
}

void vine_accel_del_vaccel(vine_accel_s *accel, vine_vaccel_s *vaccel)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    vine_assert_obj(vaccel->phys, VINE_TYPE_PHYS_ACCEL);
    vine_assert(vaccel->phys == accel);

    utils_spinlock_lock(&(vaccel->lock));
    async_condition_lock(&(accel->lock));

    int tasks = vine_vaccel_queue_size(vaccel);

    if (tasks) {
        accel->tasks -= (tasks - 1);
        async_condition_notify(&(accel->lock));
    }
    utils_list_del(&(accel->vaccels), &(vaccel->vaccels));
    vaccel->phys = 0;

    async_condition_unlock(&(accel->lock));
    vine_accel_inc_revision(accel);
    utils_spinlock_unlock(&(vaccel->lock));
}

VINE_OBJ_DTOR_DECL(vine_accel_s)
{
    vine_assert_obj(accel, VINE_TYPE_PHYS_ACCEL);
    vine_accel_s *accel = (vine_accel_s *) obj;

    async_condition_lock(&(accel->lock));
    if (accel->vaccels.length) {
        fprintf(stderr, "Erasing physical accelerator %s "
          "with %lu attached virtual accelerators!\n",
          accel->obj.name, accel->vaccels.length);
        vine_assert("Erasing physical accelerator with dangling virtual accels");
    }
    async_condition_unlock(&(accel->lock));
    vine_accel_release((vine_accel **) (&accel->free_vaq));
}
