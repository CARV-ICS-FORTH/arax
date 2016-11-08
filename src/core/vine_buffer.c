#include "vine_buffer.h"
#include "utils/system.h"

void vine_buffer_init(vine_buffer_s * buffer,void * user_buffer,size_t user_buffer_size,void * vine_data,int copy)
{
	buffer->user_buffer = user_buffer;
	buffer->user_buffer_size = user_buffer_size;
	buffer->vine_data = vine_data;
	if(copy)
		memcpy(vine_data_deref(vine_data),user_buffer,user_buffer_size);
}

int vine_buffer_compare(const void * a,const void * b)
{
	const vine_buffer_s * va = a;
	const vine_buffer_s * vb = b;
	return system_compare_ptrs(va->vine_data,vb->vine_data);
}
