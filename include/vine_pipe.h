#ifndef VINE_PIPE_HEADER
#define VINE_PIPE_HEADER
#include "vine_talk.h"
#include "structs/list.h"
#include "structs/queue.h"
#include "core/vine_accel.h"
#include "core/vine_proc.h"
#include "core/vine_data.h"
#include "core/alloc.h"

#ifdef __cplusplus
extern "C"
{
#endif
/**
 * Vineyard Task message.
 */
typedef struct vine_task_msg
{
	vine_accel * accel;	/**< Accelerator responsible for this task */
	vine_proc * proc;	/**< Process id */
	vine_data * args;	/**< Packed process arguments */
	int in_count;    	/**< Number of input buffers */
	int out_count;   	/**< Number of output buffers */
	vine_task_state_e state;
	vine_data * io[]; 	/**< in_count+out_count pointers
							to input and output buffers*/
}vine_task_msg_s;

/**
 * Shared Memory segment layout
 */
typedef struct vine_pipe
{
	void * self;					/**< Pointer to myself */
	uint64_t mapped;				/**< Current map counter  */
	structs_list_s accelerator_list;	/**< List of accelerators */
	structs_list_s process_list;		/**< List of processes */
	queue_s * queue;				/**< Queue */
	vine_alloc_s allocator;			/**< Allocator for this shared memory */
}vine_pipe_s;

/**
 * Initialize a vine_pipe.
 */
vine_pipe_s * vine_pipe_init(void * mem,size_t size,size_t ring_size);

int vine_pipe_register_accel(vine_pipe_s * pipe,vine_accel_s * accel);

vine_accel_s * vine_proc_find_accel(vine_pipe_s * pipe,const char * name,vine_accel_type_e type);

int vine_pipe_register_proc(vine_pipe_s * pipe,vine_proc_s * proc);

vine_proc_s * vine_proc_find_proc(vine_pipe_s * pipe,const char * name,vine_accel_type_e type);
/**
 * Destroy vine_pipe.
 */
int vine_pipe_exit(vine_pipe_s * pipe);

#ifdef MMAP_FIXED
#define pointer_to_offset(TYPE,BASE,VD) ((TYPE)((void*)(VD) - (void*)(BASE)))
#define offset_to_pointer(TYPE,BASE,VD) ((TYPE)((char*)(BASE)+(size_t)(VD)))
#else
#define pointer_to_offset(TYPE,BASE,VD) ((TYPE)VD)
#define offset_to_pointer(TYPE,BASE,VD) ((TYPE)VD)
#endif

#ifdef __cplusplus
}
#endif

#endif
