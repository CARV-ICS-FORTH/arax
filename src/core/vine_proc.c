#include "vine_proc.h"
#include <string.h>

vine_proc_s* vine_proc_init(vine_object_repo_s *repo, const char *name)
{
    vine_proc_s *proc =
      (vine_proc_s *) vine_object_register(repo, VINE_TYPE_PROC, name,
        sizeof(vine_proc_s), 1);

    if (!proc)
        return 0;

    return proc;
}

VineFunctor* vine_proc_get_functor(vine_proc_s *proc, vine_accel_type_e type)
{
    vine_assert(vine_accel_valid_type(type));
    return proc->functors[type];
}

VineFunctor* vine_proc_set_functor(vine_proc_s *proc, vine_accel_type_e type, VineFunctor *vfn)
{
    VineFunctor *ret = vine_proc_get_functor(proc, type);

    proc->functors[type] = vfn;
    return ret;
}

VINE_OBJ_DTOR_DECL(vine_proc_s)
{ }
