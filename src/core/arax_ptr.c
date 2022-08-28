#include "arax_ptr.h"
#include "arax_pipe.h"

int arax_ptr_valid(const void *ptr)
{
    arax_pipe_s *pipe = arax_init();
    void *vp = pipe;

    if (ptr < vp)  // Before segment start
        return 0;  // not valid

    vp += pipe->shm_size; // Move to end of segment

    if (ptr < vp)  // Before segment end
        return 1;  // valid

    return 0; // Not valid
}
