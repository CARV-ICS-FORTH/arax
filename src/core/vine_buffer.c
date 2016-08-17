#include "vine_buffer.h"

void vine_buffer_init(vine_buffer_s * buffer,void * user_buffer,size_t user_buffer_size,void * vine_data)
{
	buffer->user_buffer = user_buffer;
	buffer->user_buffer_size = user_buffer_size;
	buffer->vine_data = vine_data;
	memcpy(vine_data_deref(vine_data),user_buffer,user_buffer_size);
}
