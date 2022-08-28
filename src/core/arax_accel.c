#include "arax_pipe.h"
#include <string.h>
#include <stdlib.h>

arax_accel_s* arax_accel_init(arax_pipe_s *pipe, const char *name,
  arax_accel_type_e type, size_t size, size_t capacity)
{
    arax_accel_s *obj = (arax_accel_s *) arax_object_register(&(pipe->objs),
        ARAX_TYPE_PHYS_ACCEL,
        name, sizeof(arax_accel_s), 1);

    if (!obj)        // GCOV_EXCL_LINE
        return obj;  // GCOV_EXCL_LINE

    async_condition_init(&(pipe->async), &(obj->lock));
    obj->tasks = 0;
    utils_list_init(&(obj->vaccels));
    obj->type     = type;
    obj->state    = accel_idle;
    obj->revision = 0;
    arax_throttle_init(&(pipe->async), &(obj->throttle), size, capacity);
    obj->free_vaq = arax_vaccel_init(pipe, name, type, obj);
    return obj;
}

void arax_accel_wait_for_task(arax_accel_s *accel)
{
    async_condition_lock(&(accel->lock));

    while (accel->tasks == 0)
        async_condition_wait(&(accel->lock));

    accel->tasks--;

    async_condition_unlock(&(accel->lock));
}

void arax_accel_add_task(arax_accel_s *accel)
{
    async_condition_lock(&(accel->lock));
    accel->tasks++;
    async_condition_notify(&(accel->lock));
    async_condition_unlock(&(accel->lock));
}

size_t arax_accel_pending_tasks(arax_accel_s *accel)
{
    return accel->tasks;
}

void ARAX_THROTTLE_DEBUG_ACCEL_FUNC(arax_accel_size_inc)(arax_accel * accel,
  size_t sz ARAX_THROTTLE_DEBUG_ACCEL_PARAMS){
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_accel_s *phys = accel;

    arax_throttle_size_inc(&(phys->throttle), sz);
}

void ARAX_THROTTLE_DEBUG_ACCEL_FUNC(arax_accel_size_dec)(arax_accel * accel,
  size_t sz ARAX_THROTTLE_DEBUG_ACCEL_PARAMS){
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_accel_s *phys = accel;

    arax_throttle_size_dec(&(phys->throttle), sz);
}

size_t arax_accel_get_available_size(arax_accel *accel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_accel_s *phys = accel;

    return arax_throttle_get_available_size(&(phys->throttle));
}

size_t arax_accel_get_total_size(arax_accel *accel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_accel_s *phys = accel;

    return arax_throttle_get_total_size(&(phys->throttle));
}

const char* arax_accel_get_name(arax_accel_s *accel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    return accel->obj.name;
}

arax_accel_state_e arax_accel_get_stat(arax_accel_s *accel, arax_accel_stats_s *stat)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    /* TODO: IMPLEMENT stat memcpy */
    return accel->state;
}

void arax_accel_inc_revision(arax_accel_s *accel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    __sync_fetch_and_add(&(accel->revision), 1);
}

size_t arax_accel_get_revision(arax_accel_s *accel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    return accel->revision;
}

void arax_accel_add_vaccel(arax_accel_s *accel, arax_vaccel_s *vaccel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);

    if ( (vaccel->phys) == accel)
        return;

    arax_assert(vaccel->phys == 0);

    arax_pipe_remove_orphan_vaccel(vaccel->obj.repo->pipe, vaccel);

    utils_spinlock_lock(&(vaccel->lock));

    if ( (vaccel->phys) != accel) {
        async_condition_lock(&(accel->lock));

        utils_list_add(&(accel->vaccels), &(vaccel->vaccels));

        int tasks = arax_vaccel_queue_size(vaccel);

        if (tasks) {
            accel->tasks += tasks;
            async_condition_notify(&(accel->lock));
        }

        vaccel->phys = accel;

        async_condition_unlock(&(accel->lock));
        arax_accel_inc_revision(accel);
    }

    utils_spinlock_unlock(&(vaccel->lock));
} /* arax_accel_add_vaccel */

size_t arax_accel_get_assigned_vaccels(arax_accel_s *accel, arax_vaccel_s ***vaccel)
{
    size_t count = 0;

    async_condition_lock(&(accel->lock));
    count   = accel->vaccels.length;
    *vaccel = malloc(sizeof(arax_vaccel_s *) * count);
    utils_list_to_array(&(accel->vaccels), (void **) *vaccel);
    async_condition_unlock(&(accel->lock));

    return count;
}

void arax_accel_del_vaccel(arax_accel_s *accel, arax_vaccel_s *vaccel)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    arax_assert_obj(vaccel->phys, ARAX_TYPE_PHYS_ACCEL);
    arax_assert(vaccel->phys == accel);

    utils_spinlock_lock(&(vaccel->lock));
    async_condition_lock(&(accel->lock));

    int tasks = arax_vaccel_queue_size(vaccel);

    if (tasks) {
        accel->tasks -= (tasks - 1);
        async_condition_notify(&(accel->lock));
    }
    utils_list_del(&(accel->vaccels), &(vaccel->vaccels));
    vaccel->phys = 0;

    async_condition_unlock(&(accel->lock));
    arax_accel_inc_revision(accel);
    utils_spinlock_unlock(&(vaccel->lock));
}

ARAX_OBJ_DTOR_DECL(arax_accel_s)
{
    arax_assert_obj(accel, ARAX_TYPE_PHYS_ACCEL);
    arax_accel_s *accel = (arax_accel_s *) obj;

    async_condition_lock(&(accel->lock));
    if (accel->vaccels.length) {
        fprintf(stderr, "Erasing physical accelerator %s "
          "with %lu attached virtual accelerators!\n",
          accel->obj.name, accel->vaccels.length);
        arax_assert("Erasing physical accelerator with dangling virtual accels");
    }
    async_condition_unlock(&(accel->lock));
    arax_accel_release((arax_accel **) (&accel->free_vaq));
}
