#include "vt_plot.h"
#include "vine_pipe.h"
#include "vine_ptr.h"
#include "../alloc/alloc.h"
#include <string.h>

uint64_t* vine_plot_register_metric(const char *name, uint64_t *metric)
{
    vine_pipe_s *vpipe = vine_talk_init();

    if (metric)
        vine_assert(vine_ptr_valid(metric));
    else
        metric = arch_alloc_allocate(&(vpipe->allocator), sizeof(*metric));

    char *vname = arch_alloc_allocate(&(vpipe->allocator), strlen(name) + 1);

    strcpy(vname, name);

    utils_kv_set(&(vpipe->metrics_kv), vname, metric);

    return metric;
}
