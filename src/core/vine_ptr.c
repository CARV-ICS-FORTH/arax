#include "vine_ptr.h"
#include "vine_pipe.h"

int vine_ptr_valid(void * ptr)
{
	vine_pipe_s * pipe = vine_talk_init();
	void * vp = pipe;

	if(ptr < vp) // Before segment start
		return 0; // not valid

	vp += pipe->shm_size; // Move to end of segment

	if(ptr < vp)	// Before segment end
		return 1; // valid

	return 0; // Not valid
}
