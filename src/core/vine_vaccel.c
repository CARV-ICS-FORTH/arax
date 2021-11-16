#include "vine_pipe.h"

vine_vaccel_s* vine_vaccel_init(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type, vine_accel_s *accel)
{
    vine_vaccel_s *vaccel = (vine_vaccel_s *)
      vine_object_register(&(pipe->objs), VINE_TYPE_VIRT_ACCEL, name, sizeof(vine_vaccel_s), 1);

    if (!vaccel)   // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    vaccel->phys     = 0;
    vaccel->cid      = (uint64_t) -1;
    vaccel->priority = (uint64_t) -1;
    utils_spinlock_init(&(vaccel->lock) );
    if (!utils_queue_init(&(vaccel->queue) ) ) // GCOV_EXCL_LINE
        return 0;                              // GCOV_EXCL_LINE

    utils_list_node_init(&(vaccel->vaccels), vaccel);
    vaccel->type     = type;
    vaccel->meta     = 0;
    vaccel->ordering = SEQUENTIAL;

    if (accel)
        vine_accel_add_vaccel(accel, vaccel);
    else
        vine_pipe_add_orphan_vaccel(pipe, vaccel);

    return vaccel;
}

void vine_vaccel_add_task(vine_vaccel_s *accel, vine_task *task)
{
    utils_spinlock_lock(&(accel->lock));
    while (!utils_queue_push(&(accel->queue), task));
    if (accel->phys)
        vine_accel_add_task(accel->phys);
    utils_spinlock_unlock(&(accel->lock));
}

void vine_vaccel_set_ordering(vine_accel_s *accel, vine_accel_ordering_e ordering)
{
    vine_assert_obj(accel, VINE_TYPE_VIRT_ACCEL);
    vine_vaccel_s *vaccel = (vine_vaccel_s *) accel;

    vaccel->ordering = ordering;
}

vine_accel_ordering_e vine_vaccel_get_ordering(vine_accel_s *accel)
{
    vine_assert_obj(accel, VINE_TYPE_VIRT_ACCEL);
    vine_vaccel_s *vaccel = (vine_vaccel_s *) accel;

    return vaccel->ordering;
}

uint64_t vine_vaccel_set_cid(vine_vaccel_s *vaccel, uint64_t cid)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    vaccel->cid = cid;
    return vaccel->cid;
}

uint64_t vine_vaccel_get_cid(vine_vaccel_s *vaccel)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    return vaccel->cid;
}

uint64_t vine_vaccel_set_job_priority(vine_vaccel_s *vaccel, uint64_t priority)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    vaccel->priority = priority;
    return vaccel->priority;
}

uint64_t vine_vaccel_get_job_priority(vine_vaccel_s *vaccel)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    return vaccel->priority;
}

void vine_vaccel_set_meta(vine_vaccel_s *vaccel, void *meta)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    vaccel->meta = meta;
}

void* vine_vaccel_get_meta(vine_vaccel_s *vaccel)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    return vaccel->meta;
}

utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    return &(vaccel->queue);
}

unsigned int vine_vaccel_queue_size(vine_vaccel_s *vaccel)
{
    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);
    return utils_queue_used_slots(vine_vaccel_queue(vaccel));
}

vine_accel_state_e vine_vaccel_get_stat(vine_vaccel_s *accel, vine_accel_stats_s *stat)
{
    vine_assert_obj(accel, VINE_TYPE_VIRT_ACCEL);
    return vine_accel_get_stat(accel->phys, stat);
}

VINE_OBJ_DTOR_DECL(vine_vaccel_s)
{
    vine_vaccel_s *vaccel = (vine_vaccel_s *) obj;

    vine_assert_obj(vaccel, VINE_TYPE_VIRT_ACCEL);

    if (vaccel->phys)
        vine_accel_del_vaccel(vaccel->phys, vaccel);
    else
        vine_pipe_remove_orphan_vaccel(pipe, vaccel);
}
