#ifndef ASYNC_HEADER
#define ASYNC_HEADER
#include "core/vine_object.h"

typedef struct
{
#ifndef __cplusplus
	char padd;
#endif
}async_meta_s;

typedef struct
{
	volatile size_t completed;
}
async_completion_s;

typedef struct
{
	volatile size_t value;
}
async_semaphore_s;

#include "async_api.h"
#endif
