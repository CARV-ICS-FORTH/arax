#include "arax_proc.h"
#include <string.h>

arax_proc_s* arax_proc_init(arax_object_repo_s *repo, const char *name)
{
    arax_proc_s *proc =
      (arax_proc_s *) arax_object_register(repo, ARAX_TYPE_PROC, name,
        sizeof(arax_proc_s), 1);

    if (!proc)     // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    return proc;
}

AraxFunctor* arax_proc_get_functor(arax_proc_s *proc, arax_accel_type_e type)
{
    arax_assert(arax_accel_valid_type(type));
    return proc->functors[type];
}

AraxFunctor* arax_proc_set_functor(arax_proc_s *proc, arax_accel_type_e type, AraxFunctor *vfn)
{
    AraxFunctor *ret = arax_proc_get_functor(proc, type);

    proc->functors[type] = vfn;

    proc->canrun |= (1llu << type);

    return ret;
}

int arax_proc_can_run_at(arax_proc_s *proc, arax_accel_type_e type)
{
    arax_assert(arax_accel_valid_type(type));
    arax_assert(type != ANY);

    return !!((proc->canrun) & (1llu << type));
}

ARAX_OBJ_DTOR_DECL(arax_proc_s)
{ }
