#include <arax_pipe.h>
#include <stdio.h>
#include <string.h>

arax_pipe_s* arax_pipe_init(void *mem, size_t size, int enforce_version)
{
    arax_pipe_s *pipe = mem;
    uint64_t value;

    value = arax_pipe_add_process(pipe);

    if (value) { // Not first so assume initialized
        while (!pipe->sha[0]);
        if (strcmp(pipe->sha, ARAX_GIT_REV)) {
            fprintf(stderr, "Arax revision mismatch(%s vs %s)!", ARAX_GIT_REV, pipe->sha);
            if (enforce_version)
                return 0;
        }
        arch_alloc_init_always(&(pipe->allocator));
        return pipe;
    }

    printf("Initializing pipe.\n");

    pipe->shm_size = size;

    /**
     * Write sha sum except first byte
     */
    snprintf(pipe->sha + 1, ARAX_PIPE_SHA_SIZE - 1, "%s", &ARAX_GIT_REV[1]);
    pipe->sha[0] = ARAX_GIT_REV[0];

    arax_object_repo_init(&(pipe->objs), pipe);

    size -= sizeof(*pipe); // Subtract header size

    if (arch_alloc_init_once(&(pipe->allocator), size))
        return 0;

    if (!utils_list_init(&(pipe->orphan_vacs)) )
        return 0;

    async_meta_init_once(&(pipe->async), &(pipe->allocator) );

    async_condition_init(&(pipe->async), &(pipe->cntrl_ready_cond));

    async_condition_init(&(pipe->async), &(pipe->orphan_cond));

    arax_throttle_init(&(pipe->async), &(pipe->throttle), size, size);

    utils_kv_init(&(pipe->ass_kv));

    utils_kv_init(&(pipe->metrics_kv));

    return pipe;
} /* arax_pipe_init */

const char* arax_pipe_get_revision(arax_pipe_s *pipe)
{
    return pipe->sha;
}

void arax_pipe_add_orphan_vaccel(arax_pipe_s *pipe, arax_vaccel_s *vac)
{
    arax_assert_obj(vac, ARAX_TYPE_VIRT_ACCEL);
    arax_assert(vac->phys == 0);
    async_condition_lock(&(pipe->orphan_cond));
    utils_list_add(&(pipe->orphan_vacs), &(vac->vaccels));
    async_condition_notify(&(pipe->orphan_cond));
    async_condition_unlock(&(pipe->orphan_cond));
}

int arax_pipe_have_orphan_vaccels(arax_pipe_s *pipe)
{
    return pipe->orphan_vacs.length;
}

arax_vaccel_s* arax_pipe_get_orphan_vaccel(arax_pipe_s *pipe)
{
    arax_vaccel_s *vac = 0;

    async_condition_lock(&(pipe->orphan_cond));

    if (!arax_pipe_have_orphan_vaccels(pipe))
        async_condition_wait(&(pipe->orphan_cond));

    utils_list_node_s *lvac = utils_list_pop_head(&(pipe->orphan_vacs));

    async_condition_unlock(&(pipe->orphan_cond));

    if (lvac) {
        vac = lvac->owner;
        arax_assert_obj(vac, ARAX_TYPE_VIRT_ACCEL);
        arax_assert(vac->phys == 0);
    }

    return vac;
}

void arax_pipe_remove_orphan_vaccel(arax_pipe_s *pipe, arax_vaccel_s *vac)
{
    arax_assert_obj(vac, ARAX_TYPE_VIRT_ACCEL);
    async_condition_lock(&(pipe->orphan_cond));
    if (utils_list_node_linked(&(vac->vaccels)))
        utils_list_del(&(pipe->orphan_vacs), &(vac->vaccels));
    async_condition_unlock(&(pipe->orphan_cond));
}

void arax_pipe_orphan_stop(arax_pipe_s *pipe)
{
    async_condition_lock(&(pipe->orphan_cond));
    async_condition_notify(&(pipe->orphan_cond));
    async_condition_unlock(&(pipe->orphan_cond));
}

uint64_t arax_pipe_add_process(arax_pipe_s *pipe)
{
    return __sync_fetch_and_add(&(pipe->processes), 1);
}

uint64_t arax_pipe_del_process(arax_pipe_s *pipe)
{
    return __sync_fetch_and_add(&(pipe->processes), -1);
}

int arax_pipe_have_to_mmap(arax_pipe_s *pipe, int pid)
{
    arax_assert(pid); // Pid cant be 0
    int have_to_mmap = 1;
    int c;

    utils_spinlock_lock(&(pipe->proc_lock));
    for (c = 0; c < ARAX_PROC_MAP_SIZE; c++) {
        if (pipe->proc_map[c] == pid) {
            have_to_mmap = 0; // Already mmaped
            break;
        }

        if (!pipe->proc_map[c]) {    // Reached an unused cell
            pipe->proc_map[c] = pid; // Register new pid
            break;
        }
    }
    utils_spinlock_unlock(&(pipe->proc_lock));
    return have_to_mmap;
}

void arax_pipe_mark_unmap(arax_pipe_s *pipe, int pid)
{
    arax_assert(pid); // Pid cant be 0
    int c;

    utils_spinlock_lock(&(pipe->proc_lock));
    for (c = 0; c < ARAX_PROC_MAP_SIZE; c++) {
        if (pipe->proc_map[c] == pid) // Found PID location
            break;
    }
    arax_assert(c < ARAX_PROC_MAP_SIZE); // pid should be in the proc_map
    // Skip cell containing pid
    memmove(pipe->proc_map + c, pipe->proc_map + (c + 1), ARAX_PROC_MAP_SIZE - (c + 1));
    pipe->proc_map[ARAX_PROC_MAP_SIZE - 1] = 0;
    utils_spinlock_unlock(&(pipe->proc_lock));
}

void* arax_pipe_mmap_address(arax_pipe_s *pipe)
{
    int value = __sync_bool_compare_and_swap(&(pipe->self), 0, pipe);

    if (value)
        return pipe;
    else
        return pipe->self;
}

int arax_pipe_delete_accel(arax_pipe_s *pipe, arax_accel_s *accel)
{
    utils_list_node_s *itr;
    utils_list_s *list;
    arax_accel_s *accel_in_list = 0;

    list = arax_object_list_lock(&(pipe->objs), ARAX_TYPE_PHYS_ACCEL);
    utils_list_for_each(*list, itr){
        accel_in_list = (arax_accel_s *) itr->owner;
        if (accel == accel_in_list) {
            arax_object_list_unlock(&(pipe->objs), ARAX_TYPE_PHYS_ACCEL);
            arax_object_ref_dec(&(accel->obj));
            return 0;
        }
    }
    arax_object_list_unlock(&(pipe->objs), ARAX_TYPE_PHYS_ACCEL);
    return 1;
}

arax_accel_s* arax_pipe_find_accel(arax_pipe_s *pipe, const char *name,
  arax_accel_type_e type)
{
    utils_list_node_s *itr;
    utils_list_s *list;
    arax_accel_s *accel = 0;

    list = arax_object_list_lock(&(pipe->objs), ARAX_TYPE_PHYS_ACCEL);
    utils_list_for_each(*list, itr){
        accel = (arax_accel_s *) itr->owner;
        if (type && (type != accel->type) )
            continue;
        if (!name ||
          (strcmp(name, arax_accel_get_name(accel) ) == 0) )
        {
            arax_object_list_unlock(&(pipe->objs),
              ARAX_TYPE_PHYS_ACCEL);
            return accel;
        }
    }
    accel = 0;
    arax_object_list_unlock(&(pipe->objs), ARAX_TYPE_PHYS_ACCEL);
    return accel;
}

arax_proc_s* arax_pipe_find_proc(arax_pipe_s *pipe, const char *name)
{
    utils_list_node_s *itr;
    utils_list_s *list;
    arax_proc_s *proc;

    list = arax_object_list_lock(&(pipe->objs), ARAX_TYPE_PROC);
    utils_list_for_each(*list, itr){
        proc = (arax_proc_s *) itr->owner;
        if (strcmp(name, proc->obj.name) == 0) {
            arax_object_list_unlock(&(pipe->objs), ARAX_TYPE_PROC);
            return proc;
        }
    }
    proc = 0;
    arax_object_list_unlock(&(pipe->objs), ARAX_TYPE_PROC);
    return proc;
}

/**
 * Destroy arax_pipe.
 */
int arax_pipe_exit(arax_pipe_s *pipe)
{
    int ret = arax_pipe_del_process(pipe) == 1;

    if (ret) { // Last user
        arax_object_repo_exit(&(pipe->objs));
        async_meta_exit(&(pipe->async) );
        arch_alloc_exit(&(pipe->allocator) );
    }
    return ret;
}

void ARAX_PIPE_THOTTLE_DEBUG_FUNC(arax_pipe_size_inc)(arax_pipe_s * pipe, size_t sz ARAX_PIPE_THOTTLE_DEBUG_PARAMS){
    arax_assert(pipe);
    arax_throttle_size_inc(&pipe->throttle, sz);
}


void ARAX_PIPE_THOTTLE_DEBUG_FUNC(arax_pipe_size_dec)(arax_pipe_s * pipe, size_t sz ARAX_PIPE_THOTTLE_DEBUG_PARAMS){
    arax_assert(pipe);
    arax_throttle_size_dec(&pipe->throttle, sz);
}


size_t arax_pipe_get_available_size(arax_pipe_s *pipe)
{
    arax_assert(pipe);
    return arax_throttle_get_available_size(&pipe->throttle);
}

size_t arax_pipe_get_total_size(arax_pipe_s *pipe)
{
    arax_assert(pipe);
    return arax_throttle_get_total_size(&pipe->throttle);
}
