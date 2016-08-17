#ifndef VINE_BUFFER_HEADER
#define VINE_BUFFER_HEADER

#include <string.h>

typedef struct
{
	void * user_buffer;
	size_t user_buffer_size;
	void * vine_data;
}vine_buffer_s;

#include "core/vine_data.h"

/**
 * Define a Vineyard buffer.
 * @param USER_POINTER Pointer to a valid memory location used for data.
 * @param BUFFER_SIZE Size of memory pointer by USER_POINTER in bytes.
 */
#define VINE_BUFFER(USER_POINTER,BUFFER_SIZE)								\
{.user_buffer = USER_POINTER,.user_buffer_size = BUFFER_SIZE,.vine_data = 0}

void vine_buffer_init(vine_buffer_s * buffer,void * user_buffer,size_t user_buffer_size,void * vine_data);

#endif
