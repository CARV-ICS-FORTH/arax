#ifndef VINE_PROC_HEADER
#define VINE_PROC_HEADER
#include <vine_talk.h>
#include "core/vine_object.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct
{
    vine_object_s obj;
    uint64_t      canrun;                 // < One bit set for each VINE_ACCEL_TYPE that has a functor
    static_assert(VINE_ACCEL_TYPES < 64); // More accel types, than can fit in canrun
    VineFunctor * functors[VINE_ACCEL_TYPES];
    /* To add more as needed */
} vine_proc_s;

/**
 * Initialize a vine_proc at the memory pointed by \c mem.
 *
 * @param repo The vine_object_repo_s that will track the initialized procedure.
 * @param name NULL terminated string, will be copied to private buffer.
 * @return An initialized instance of vine_proc_s, NULL on failure.
 */
vine_proc_s* vine_proc_init(vine_object_repo_s *repo, const char *name);

/**
 * Return \c proc functor pointer for provided \c type.
 * @return Pointer to functor, null is returned if no functior is set for given \c type.
 */
VineFunctor* vine_proc_get_functor(vine_proc_s *proc, vine_accel_type_e type);

/**
 * Set the VineFunctor of \c proc for the provided \c type.
 * @param proc An initialized vine_proc_s instance.
 * @param type Accelerator type for provided functor.
 * @param vfn Functor pointer, can be null.
 * @return Returns previous value of \c proc, just as vine_proc_get_functor() would return.
 */
VineFunctor* vine_proc_set_functor(vine_proc_s *proc, vine_accel_type_e type, VineFunctor *vfn);

/**
 * Returns if \c proc can 'run' in an accelerator of \c type.
 *
 * \note \c type has to be different than ANY.
 *
 * @param proc An initialized vine_proc_s instance.
 * @param type Accelerator type to check against.
 * @return 0 if \c type can not execute \c proc. Non zero otherwise.
 */
int vine_proc_can_run_at(vine_proc_s *proc, vine_accel_type_e type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_PROC_HEADER */
