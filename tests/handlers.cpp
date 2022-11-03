#include "arax.h"
#include "core/arax_task.h" // TODO: Remove dependency to private header

ARAX_HANDLER_EX(test_handler_ex, GPU, int *upa)
{
    // Test user provided arguement
    upa = 0;
    return task_completed;
}

ARAX_HANDLER(test_handler, GPU)
{
    return task_completed;
}
