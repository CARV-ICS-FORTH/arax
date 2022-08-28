#ifndef ARAX_PROC_HEADER
#define ARAX_PROC_HEADER
#include <arax.h>
#include "core/arax_object.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

#if ARAX_ACCEL_TYPES > 64
#error More accel types, than can fit in arax_proc_s.canrun
#endif


typedef struct
{
    arax_object_s obj;
    uint64_t      canrun; // < One bit set for each ARAX_ACCEL_TYPE that has a functor
    AraxFunctor * functors[ARAX_ACCEL_TYPES];
    /* To add more as needed */
} arax_proc_s;

/**
 * Initialize a arax_proc at the memory pointed by \c mem.
 *
 * @param repo The arax_object_repo_s that will track the initialized procedure.
 * @param name NULL terminated string, will be copied to private buffer.
 * @return An initialized instance of arax_proc_s, NULL on failure.
 */
arax_proc_s* arax_proc_init(arax_object_repo_s *repo, const char *name);

/**
 * Return \c proc functor pointer for provided \c type.
 * @return Pointer to functor, null is returned if no functior is set for given \c type.
 */
AraxFunctor* arax_proc_get_functor(arax_proc_s *proc, arax_accel_type_e type);

/**
 * Set the AraxFunctor of \c proc for the provided \c type.
 * @param proc An initialized arax_proc_s instance.
 * @param type Accelerator type for provided functor.
 * @param vfn Functor pointer, can be null.
 * @return Returns previous value of \c proc, just as arax_proc_get_functor() would return.
 */
AraxFunctor* arax_proc_set_functor(arax_proc_s *proc, arax_accel_type_e type, AraxFunctor *vfn);

/**
 * Returns if \c proc can 'run' in an accelerator of \c type.
 *
 * \note \c type has to be different than ANY.
 *
 * @param proc An initialized arax_proc_s instance.
 * @param type Accelerator type to check against.
 * @return 0 if \c type can not execute \c proc. Non zero otherwise.
 */
int arax_proc_can_run_at(arax_proc_s *proc, arax_accel_type_e type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_PROC_HEADER */
