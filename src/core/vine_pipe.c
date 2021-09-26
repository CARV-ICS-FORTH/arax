#include <vine_pipe.h>
#include <stdio.h>
#include <string.h>

vine_pipe_s* vine_pipe_init(void *mem, size_t size, int enforce_version)
{
    vine_pipe_s *pipe = mem;
    uint64_t value;

    value = vine_pipe_add_process(pipe);

    if (value) { // Not first so assume initialized
        while (!pipe->sha[0]);
        if (strcmp(pipe->sha, VINE_TALK_GIT_REV)) {
            fprintf(stderr, "Vinetalk revision mismatch(%s vs %s)!", VINE_TALK_GIT_REV, pipe->sha);
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
    snprintf(pipe->sha + 1, VINE_PIPE_SHA_SIZE - 1, "%s", VINE_TALK_GIT_REV + 1);
    pipe->sha[0] = VINE_TALK_GIT_REV[0];

    vine_object_repo_init(&(pipe->objs), pipe);

    if (arch_alloc_init_once(&(pipe->allocator), size - sizeof(*pipe) ))
        return 0;

    if (!utils_list_init(&(pipe->orphan_vacs)) )
        return 0;

    async_meta_init_once(&(pipe->async), &(pipe->allocator) );

    async_condition_init(&(pipe->async), &(pipe->orphan_cond));

    vine_throttle_init(&(pipe->async), &(pipe->throttle), size, size);

    utils_kv_init(&(pipe->ass_kv));

    utils_kv_init(&(pipe->metrics_kv));

    return pipe;
} /* vine_pipe_init */

const char* vine_pipe_get_revision(vine_pipe_s *pipe)
{
    return pipe->sha;
}

void vine_pipe_add_orphan_vaccel(vine_pipe_s *pipe, vine_vaccel_s *vac)
{
    vine_assert_obj(vac, VINE_TYPE_VIRT_ACCEL);
    vine_assert(vac->phys == 0);
    async_condition_lock(&(pipe->orphan_cond));
    utils_list_add(&(pipe->orphan_vacs), &(vac->vaccels));
    async_condition_notify(&(pipe->orphan_cond));
    async_condition_unlock(&(pipe->orphan_cond));
}

int vine_pipe_have_orphan_vaccels(vine_pipe_s *pipe)
{
    return pipe->orphan_vacs.length;
}

vine_vaccel_s* vine_pipe_get_orphan_vaccel(vine_pipe_s *pipe)
{
    vine_vaccel_s *vac = 0;

    async_condition_lock(&(pipe->orphan_cond));

    if (pipe->orphan_vacs.length == 0)
        async_condition_wait(&(pipe->orphan_cond));

    utils_list_node_s *lvac = utils_list_pop_head(&(pipe->orphan_vacs));

    async_condition_unlock(&(pipe->orphan_cond));

    if (lvac) {
        vac = lvac->owner;
        vine_assert_obj(vac, VINE_TYPE_VIRT_ACCEL);
        vine_assert(vac->phys == 0);
    }

    return vac;
}

void vine_pipe_remove_orphan_vaccel(vine_pipe_s *pipe, vine_vaccel_s *vac)
{
    vine_assert_obj(vac, VINE_TYPE_VIRT_ACCEL);
    async_condition_lock(&(pipe->orphan_cond));
    if (utils_list_node_linked(&(vac->vaccels)))
        utils_list_del(&(pipe->orphan_vacs), &(vac->vaccels));
    async_condition_unlock(&(pipe->orphan_cond));
}

void vine_pipe_orphan_stop(vine_pipe_s *pipe)
{
    async_condition_lock(&(pipe->orphan_cond));
    async_condition_notify(&(pipe->orphan_cond));
    async_condition_unlock(&(pipe->orphan_cond));
}

uint64_t vine_pipe_add_process(vine_pipe_s *pipe)
{
    return __sync_fetch_and_add(&(pipe->processes), 1);
}

uint64_t vine_pipe_del_process(vine_pipe_s *pipe)
{
    return __sync_fetch_and_add(&(pipe->processes), -1);
}

int vine_pipe_have_to_mmap(vine_pipe_s *pipe, int pid)
{
    vine_assert(pid); // Pid cant be 0
    int have_to_mmap = 1;
    int c;

    utils_spinlock_lock(&(pipe->proc_lock));
    for (c = 0; c < VINE_PROC_MAP_SIZE; c++) {
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

void vine_pipe_mark_unmap(vine_pipe_s *pipe, int pid)
{
    vine_assert(pid); // Pid cant be 0
    int c;

    utils_spinlock_lock(&(pipe->proc_lock));
    for (c = 0; c < VINE_PROC_MAP_SIZE; c++) {
        if (pipe->proc_map[c] == pid) // Found PID location
            break;
    }
    vine_assert(c < VINE_PROC_MAP_SIZE); // pid should be in the proc_map
    // Skip cell containing pid
    memmove(pipe->proc_map + c, pipe->proc_map + (c + 1), VINE_PROC_MAP_SIZE - (c + 1));
    pipe->proc_map[VINE_PROC_MAP_SIZE - 1] = 0;
    utils_spinlock_unlock(&(pipe->proc_lock));
}

void* vine_pipe_mmap_address(vine_pipe_s *pipe)
{
    int value = __sync_bool_compare_and_swap(&(pipe->self), 0, pipe);

    if (value)
        return pipe;
    else
        return pipe->self;
}

int vine_pipe_delete_accel(vine_pipe_s *pipe, vine_accel_s *accel)
{
    utils_list_node_s *itr;
    utils_list_s *list;
    vine_accel_s *accel_in_list = 0;

    list = vine_object_list_lock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
    utils_list_for_each(*list, itr){
        accel_in_list = (vine_accel_s *) itr->owner;
        if (accel == accel_in_list) {
            vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
            vine_object_ref_dec(&(accel->obj));
            return 0;
        }
    }
    vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
    return 1;
}

vine_accel_s* vine_pipe_find_accel(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type)
{
    utils_list_node_s *itr;
    utils_list_s *list;
    vine_accel_s *accel = 0;

    list = vine_object_list_lock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
    utils_list_for_each(*list, itr){
        accel = (vine_accel_s *) itr->owner;
        if (type && (type != accel->type) )
            continue;
        if (!name ||
          (strcmp(name, vine_accel_get_name(accel) ) == 0) )
        {
            vine_object_list_unlock(&(pipe->objs),
              VINE_TYPE_PHYS_ACCEL);
            return accel;
        }
    }
    accel = 0;
    vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PHYS_ACCEL);
    return accel;
}

vine_proc_s* vine_pipe_find_proc(vine_pipe_s *pipe, const char *name)
{
    utils_list_node_s *itr;
    utils_list_s *list;
    vine_proc_s *proc;

    list = vine_object_list_lock(&(pipe->objs), VINE_TYPE_PROC);
    utils_list_for_each(*list, itr){
        proc = (vine_proc_s *) itr->owner;
        if (strcmp(name, proc->obj.name) == 0) {
            vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PROC);
            return proc;
        }
    }
    proc = 0;
    vine_object_list_unlock(&(pipe->objs), VINE_TYPE_PROC);
    return proc;
}

/**
 * Destroy vine_pipe.
 */
int vine_pipe_exit(vine_pipe_s *pipe)
{
    int ret = vine_pipe_del_process(pipe) == 1;

    if (ret) { // Last user
        vine_object_repo_exit(&(pipe->objs));
        async_meta_exit(&(pipe->async) );
        arch_alloc_exit(&(pipe->allocator) );
    }
    return ret;
}

void VINE_PIPE_THOTTLE_DEBUG_FUNC(vine_pipe_size_inc)(vine_pipe_s * pipe, size_t sz VINE_PIPE_THOTTLE_DEBUG_PARAMS){
    vine_assert(pipe);
    vine_throttle_size_inc(&pipe->throttle, sz);
}


void VINE_PIPE_THOTTLE_DEBUG_FUNC(vine_pipe_size_dec)(vine_pipe_s * pipe, size_t sz VINE_PIPE_THOTTLE_DEBUG_PARAMS){
    vine_assert(pipe);
    vine_throttle_size_dec(&pipe->throttle, sz);
}


size_t vine_pipe_get_available_size(vine_pipe_s *pipe)
{
    vine_assert(pipe);
    return vine_throttle_get_available_size(&pipe->throttle);
}

size_t vine_pipe_get_total_size(vine_pipe_s *pipe)
{
    vine_assert(pipe);
    return vine_throttle_get_total_size(&pipe->throttle);
}
