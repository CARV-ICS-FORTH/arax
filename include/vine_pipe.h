#ifndef VINE_PIPE_HEADER
#define VINE_PIPE_HEADER
#include <vine_talk.h>
#include <utils/Kv.h>
#include "utils/queue.h"
#include "core/vine_accel.h"
#include "core/vine_task.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct
{
    int pid;
    int threads;
} vine_process_tracker_s;

#define VINE_PIPE_SHA_SIZE 48

/**
 * Shared Memory segment layout
 */
typedef struct vine_pipe
{
    char               sha[VINE_PIPE_SHA_SIZE + 1];  /**< Git revision 48+1 for \0 */
    void               *self;                        /**< Pointer to myself */
    uint64_t           shm_size;                     /**< Size in bytes of shared region */
    uint64_t           processes;                    /**< Process counter - Processes using this */
    utils_spinlock     proc_lock;                    /**< Protect the proc_map */
    uint64_t           proc_map[VINE_PROC_MAP_SIZE]; /**< Array contains PIDs of all mmaped processes */
    uint64_t           last_uid;                     /**< Last instance UID */
    vine_object_repo_s objs;                         /**< Vine object repository  */
    async_meta_s       async;                        /**< Async related metadata  */
    vine_throttle_s    throttle;

    async_semaphore_s  orphan_sem;  /**< Counts \c orphan_vacs */
    utils_queue_s *    orphan_vacs; /**< Unassigned virtual accels */

    utils_kv_s         ass_kv;     /**< Assignees KV, <assigne_id,task_count>*/
    utils_kv_s         metrics_kv; /**< Name, Metric */

    arch_alloc_s       allocator; /**< Allocator for this shared memory */
} vine_pipe_s;

/**
 * Get VineTalk revision
 *
 * @param pipe vine_pipe instance.
 * @return const string with VineTalk revision.
 */
const char* vine_pipe_get_revision(vine_pipe_s *pipe);

/**
 * Add \c vac to the list of orphan_vacs/ unassigned accels.
 *
 * @param pipe vine_pipe instance.
 * @param vac Unassigned/Orphan Virtual Acceleator instance.
 */
void vine_pipe_add_orphan_vaccel(vine_pipe_s *pipe, vine_vaccel_s *vac);

/**
 * Will return != 0 if there are orphan vaccels.
 *
 * @Note: This function may return old values.
 *
 * @param pipe vine_pipe instance.
 * @return 0 if no orphans, may return any non zero value if orphans exist
 */
int vine_pipe_have_orphan_vaccels(vine_pipe_s *pipe);

/**
 * Return an orphan/unassigned virtual accelerator.
 * Function may sleep if no orphans exist at the time of the call.
 * Returned vine_vaccel_s should either be assigned to a vine_accel_s using
 * \c vine_accel_add_vaccel(), or should be marked again as orphan using
 * \c vine_pipe_add_orphan_vaccel()
 *
 * @return Unassigned/Orphan Virtual Acceleator instance.
 */
vine_vaccel_s* vine_pipe_get_orphan_vaccel(vine_pipe_s *pipe);

/**
 * Increase process counter for \c pipe.
 *
 * @param pipe vine_pipe instance.
 * @return Number of active processes before adding issuer.
 */
uint64_t vine_pipe_add_process(vine_pipe_s *pipe);

/**
 * Decrease process counter for \c pipe.
 *
 * @param pipe vine_pipe instance.
 * @return Number of active processes before removing issuer.
 */
uint64_t vine_pipe_del_process(vine_pipe_s *pipe);

/**
 * Return if we have to mmap, for the given pid.
 * This will return 1, only the first time it is callled with
 * a specific \c pid.
 */
int vine_pipe_have_to_mmap(vine_pipe_s *pipe, int pid);

/**
 * This should be called after munmap'ing \c pipe, in \c pid process.
 */
void vine_pipe_mark_unmap(vine_pipe_s *pipe, int pid);

/**
 * Return (and set if needed) the mmap location for \c pipe.
 *
 * @param pipe vine_pipe instance.
 */
void* vine_pipe_mmap_address(vine_pipe_s *pipe);

/**
 * Initialize a vine_pipe.
 *
 * \note This function must be called from all end points in order to
 * initialize a vine_pipe instance.Concurrent issuers will be serialized
 * and the returned vine_pipe instance will be initialized by the 'first'
 * issuer. All subsequent issuers will receive the already initialized
 * instance.
 *
 *
 * @param mem Shared memory pointer.
 * @param size Size of the shared memory in bytes.
 * @param enforce_version Set to 0 to make version mismatch non fatal.
 * @return An initialized vine_pipe_s instance.
 */
vine_pipe_s* vine_pipe_init(void *mem, size_t size, int enforce_version);

/**
 * Remove \c accel from the \c pipe accelerator list.
 *
 * @param pipe The pipe instance where the accelerator belongs.
 * @param accel The accelerator to be removed.
 * @return Returns 0 on success.
 */
int vine_pipe_delete_accel(vine_pipe_s *pipe, vine_accel_s *accel);

/**
 * Find an accelerator matching the user specified criteria.
 *
 * @param pipe vine_pipe instance.
 * @param name The cstring name of the accelerator, \
 *                              NULL if we dont care for the name.
 * @param type Type of the accelerator, see vine_accel_type_e.
 * @return An vine_accel_s instance, NULL on failure.
 */
vine_accel_s* vine_pipe_find_accel(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type);

/**
 * Find a procedure matching the user specified criteria.
 *
 * @param pipe vine_pipe instance.
 * @param name The cstring name of the procedure.
 * @param type Type of the procedure, see vine_accel_type_e.
 * @return An vine_proc_s instance, NULL on failure.
 */
vine_proc_s* vine_pipe_find_proc(vine_pipe_s *pipe, const char *name,
  vine_accel_type_e type);

/**
 * Destroy vine_pipe.
 *
 * \note Ensure you perform any cleanup(e.g. delete shared segment)
 * when return value becomes 0.
 *
 * @param pipe vine_pipe instance to be destroyed.
 * @return Number of remaining users of this shared segment.
 */
int vine_pipe_exit(vine_pipe_s *pipe);

#ifdef VINE_THROTTLE_DEBUG
#define VINE_PIPE_THOTTLE_DEBUG_PARAMS , const char *parent
#define VINE_PIPE_THOTTLE_DEBUG_FUNC(FUNC) __ ## FUNC
#define vine_pipe_size_inc(PIPE, SZ)       __vine_pipe_size_inc(PIPE, SZ, __func__)
#define vine_pipe_size_dec(PIPE, SZ)       __vine_pipe_size_dec(PIPE, SZ, __func__)
#else
#define VINE_PIPE_THOTTLE_DEBUG_PARAMS
#define VINE_PIPE_THOTTLE_DEBUG_FUNC(FUNC) FUNC
#endif

/**
 * Increments available size of gpu by sz
 *
 * @param pipe   pipe for shm
 * @param sz     Size of added data
 * @return       Nothing .
 */
void VINE_PIPE_THOTTLE_DEBUG_FUNC(vine_pipe_size_inc)(vine_pipe_s * pipe, size_t sz VINE_PIPE_THOTTLE_DEBUG_PARAMS);

/**
 * Decrements available size of gpu by sz
 *
 * @param pipe   pipe for shm
 * @param sz     size of removed data
 * @return       Nothing .
 */
void VINE_PIPE_THOTTLE_DEBUG_FUNC(vine_pipe_size_dec)(vine_pipe_s * pipe, size_t sz VINE_PIPE_THOTTLE_DEBUG_PARAMS);

/**
 * Gets available size of shm
 *
 * @param pipe   pipe for shm
 * @return       Avaliable size of shm
 */
size_t vine_pipe_get_available_size(vine_pipe_s *pipe);

/**
 * Gets available total size of shm
 *
 * @param pipe   pipe for shm
 * @return       Total size of shm
 */
size_t vine_pipe_get_total_size(vine_pipe_s *pipe);

#ifdef MMAP_FIXED
#define pointer_to_offset(TYPE, BASE, \
      VD)                                 ( (TYPE) ( (void *) (VD) -(void *) (BASE) ) )
#define offset_to_pointer(TYPE, BASE, \
      VD)                                 ( (TYPE) ( (char *) (BASE) + (size_t) (VD) ) )
#else /* ifdef MMAP_FIXED */
#define pointer_to_offset(TYPE, BASE, VD) ( (TYPE) VD )
#define offset_to_pointer(TYPE, BASE, VD) ( (TYPE) VD )
#endif /* ifdef MMAP_FIXED */

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_PIPE_HEADER */
