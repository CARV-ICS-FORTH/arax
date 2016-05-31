#ifndef ARCH_ASYNC_HEADER
#define ARCH_ASYNC_HEADER
#include "core/vine_object.h"

typedef struct
{
	volatile size_t counter;
}
arch_async_completion_s;

#include "../async_api.h"
#endif
