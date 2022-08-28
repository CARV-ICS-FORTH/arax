#ifndef ARAX_OBJECT_HEADER
#define ARAX_OBJECT_HEADER
#include "utils/list.h"
#include "utils/spinlock.h"
#include "arch/alloc.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Enumeration with available Arax Object Types.
 */
typedef enum arax_object_type
{
    ARAX_TYPE_PHYS_ACCEL, /* Physical Accelerator */
    ARAX_TYPE_VIRT_ACCEL, /* Virtual Accelerator */
    ARAX_TYPE_PROC,       /* Procedure */
    ARAX_TYPE_DATA,       /* Data Allocation */
    ARAX_TYPE_TASK,       /* Task */
    ARAX_TYPE_COUNT       /* Number of types */
} arax_object_type_e;

/**
 * Arax object repository struct, contains references to all Arax objects.
 */
typedef struct
{
    arax_pipe_s *pipe;
    struct
    {
        utils_list_s   list;
        utils_spinlock lock;
    } repo[ARAX_TYPE_COUNT];
} arax_object_repo_s;

/**
 * Arax Object super class, all Arax Objects have this as their first member.
 */
typedef struct
{
    arax_object_repo_s *repo;
    utils_list_node_s   list;
    size_t              alloc_size;
    arax_object_type_e  type;
    volatile int        ref_count;
    char                name[ARAX_OBJECT_NAME_SIZE];
} arax_object_s;

/**
 * Initialize an arax_object_repo_s instance on allocated pointer \c repo.
 *
 * @param repo An at least sizeof(arax_object_repo_s) big buffer.
 * @param pipe arax_pipe_s owning all objects.
 */
void arax_object_repo_init(arax_object_repo_s *repo, arax_pipe_s *pipe);

/**
 * Perform cleanup and exit time checks.
 *
 * Prints on stderr, the Objects still registered and thus considered as
 * leaks.
 *
 * @param repo A valid arax_object_repo_s instance.
 * @return Number of 'leaked' Arax Objects.
 */
int arax_object_repo_exit(arax_object_repo_s *repo);

/**
 * Convert a arax_object_type_e value to a human readable string.
 * If \c type is not a valid arax_object_type_e, then NULL is returned.
 *
 * @return Returns the string representation of \c type
 */
const char* arax_object_type_to_str(arax_object_type_e type);

/**
 * Arax Object 'Constructor'
 *
 * Register \c obj at \c repo.
 *
 * Note: Sets reference count to 1
 *
 * @param repo A valid arax_object_repo_s instance.
 * @param type Type of the new arax_object.
 * @param name The name on the new arax_object.
 * @param size The size of the new object (sizeof(struct)).
 * @param ref_count Initialize ref count
 */
arax_object_s* arax_object_register(arax_object_repo_s *repo,
  arax_object_type_e type, const char *name, size_t size, const int ref_count);

/**
 * Change name of \c obj to printf like format \c fmt
 *
 * @param obj A valid arax_object_s instance.
 * @param fmt printf style format string
 * @param ... Args matching \c fmt
 */
void arax_object_rename(arax_object_s *obj, const char *fmt, ...);

/**
 * Increase reference count of \c obj.
 *
 * @param obj A valid arax_object_s instance.
 */
void arax_object_ref_inc(arax_object_s *obj);

/**
 * Decrease reference count of \c obj.
 *
 * @param obj A valid arax_object_s instance.
 * @return Reference count after decreasing, 0 means object was reclaimed
 */
int arax_object_ref_dec(arax_object_s *obj);

/**
 * Decrease reference count of \c obj.
 * \note Assumes object repo lock is held.
 *
 * @param obj A valid arax_object_s instance.
 * @return Reference count after decreasing, 0 means object was reclaimed
 */
int arax_object_ref_dec_pre_locked(arax_object_s *obj);

/**
 * Returns \c obj current reference count.
 *
 * @param obj A valid arax_object_s instance.
 */
int arax_object_refs(arax_object_s *obj);

/**
 * Get a locked utils_list_s for traversing all objects of \c type.
 *
 * @param repo A valid arax_object_repo_s instance.
 * @param type Type of objects contained in list.
 */
utils_list_s* arax_object_list_lock(arax_object_repo_s *repo,
  arax_object_type_e                                    type);

/**
 * Unlock previously locked list.
 * Parameters should be the same with the previous arax_object_list_locked()
 * invocation.
 *
 * @param repo A valid arax_object_repo_s instance.
 * @param type Type of objects contained in list.
 */
void arax_object_list_unlock(arax_object_repo_s *repo, arax_object_type_e type);

#define ARAX_OBJ_DTOR_DECL(TYPE) void __dtor_ ## TYPE(arax_pipe_s * pipe, arax_object_s * obj)
#define ARAX_OBJ_DTOR_USE(TYPE)  __dtor_ ## TYPE

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_OBJECT_HEADER */
