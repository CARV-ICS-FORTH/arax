#ifndef ASYNC_HEADER
#define ASYNC_HEADER
#include "core/vine_object.h"

typedef struct
{
}async_meta_s;

typedef struct
{
	volatile size_t counter;
}
async_completion_s;

#include "async_api.h"
#endif
