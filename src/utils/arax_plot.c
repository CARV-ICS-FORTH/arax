#include "arax_plot.h"
#include "arax_pipe.h"
#include "arax_ptr.h"
#include "../alloc/alloc.h"
#include <string.h>

uint64_t* arax_plot_register_metric(const char *name, uint64_t *metric)
{
    arax_pipe_s *vpipe = arax_init();

    if (metric)
        arax_assert(arax_ptr_valid(metric));
    else
        metric = arch_alloc_allocate(&(vpipe->allocator), sizeof(*metric));

    char *vname = arch_alloc_allocate(&(vpipe->allocator), strlen(name) + 1);

    strcpy(vname, name);

    utils_kv_set(&(vpipe->metrics_kv), vname, metric);

    return metric;
}
