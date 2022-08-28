#include "arax_pipe.h"

arax_vaccel_s* arax_vaccel_init(arax_pipe_s *pipe, const char *name,
  arax_accel_type_e type, arax_accel_s *accel)
{
    arax_vaccel_s *vaccel = (arax_vaccel_s *)
      arax_object_register(&(pipe->objs), ARAX_TYPE_VIRT_ACCEL, name, sizeof(arax_vaccel_s), 1);

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
        arax_accel_add_vaccel(accel, vaccel);
    else
        arax_pipe_add_orphan_vaccel(pipe, vaccel);

    return vaccel;
}

void arax_vaccel_add_task(arax_vaccel_s *accel, arax_task *task)
{
    utils_spinlock_lock(&(accel->lock));
    while (!utils_queue_push(&(accel->queue), task));
    if (accel->phys)
        arax_accel_add_task(accel->phys);
    utils_spinlock_unlock(&(accel->lock));
}

void arax_vaccel_set_ordering(arax_accel_s *accel, arax_accel_ordering_e ordering)
{
    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);
    arax_vaccel_s *vaccel = (arax_vaccel_s *) accel;

    vaccel->ordering = ordering;
}

arax_accel_ordering_e arax_vaccel_get_ordering(arax_accel_s *accel)
{
    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);
    arax_vaccel_s *vaccel = (arax_vaccel_s *) accel;

    return vaccel->ordering;
}

uint64_t arax_vaccel_set_cid(arax_vaccel_s *vaccel, uint64_t cid)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    vaccel->cid = cid;
    return vaccel->cid;
}

uint64_t arax_vaccel_get_cid(arax_vaccel_s *vaccel)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    return vaccel->cid;
}

uint64_t arax_vaccel_set_job_priority(arax_vaccel_s *vaccel, uint64_t priority)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    vaccel->priority = priority;
    return vaccel->priority;
}

uint64_t arax_vaccel_get_job_priority(arax_vaccel_s *vaccel)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    return vaccel->priority;
}

void arax_vaccel_set_meta(arax_vaccel_s *vaccel, void *meta)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    vaccel->meta = meta;
}

void* arax_vaccel_get_meta(arax_vaccel_s *vaccel)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    return vaccel->meta;
}

utils_queue_s* arax_vaccel_queue(arax_vaccel_s *vaccel)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    return &(vaccel->queue);
}

unsigned int arax_vaccel_queue_size(arax_vaccel_s *vaccel)
{
    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);
    return utils_queue_used_slots(arax_vaccel_queue(vaccel));
}

arax_accel_state_e arax_vaccel_get_stat(arax_vaccel_s *accel, arax_accel_stats_s *stat)
{
    arax_assert_obj(accel, ARAX_TYPE_VIRT_ACCEL);
    return arax_accel_get_stat(accel->phys, stat);
}

ARAX_OBJ_DTOR_DECL(arax_vaccel_s)
{
    arax_vaccel_s *vaccel = (arax_vaccel_s *) obj;

    arax_assert_obj(vaccel, ARAX_TYPE_VIRT_ACCEL);

    if (vaccel->phys)
        arax_accel_del_vaccel(vaccel->phys, vaccel);
    else
        arax_pipe_remove_orphan_vaccel(pipe, vaccel);
}
