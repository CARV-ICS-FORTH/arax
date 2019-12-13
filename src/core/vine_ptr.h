#ifndef VINE_PTR_HEADER
	#define VINE_PTR_HEADER
	#include "utils/compat.h"
	/**
	 * Return true if \c ptr is 'inside' the shared segment ranges.
	 */
	VINE_CPP int vine_ptr_valid(void * ptr);
#endif
