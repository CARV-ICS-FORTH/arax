/**
 * @file
 * Example use of the VinePipe API:
 * \include consumer.c
 */

#ifndef VINE_PIPE_HEADER
#define VINE_PIPE_HEADER
#include <vine_talk.h>
#include "arch/alloc.h"
#include "utils/list.h"
#include "utils/queue.h"
#include "core/vine_accel.h"
#include "core/vine_proc.h"
#include "core/vine_data.h"

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
	uint64_t shm_size;				/**< Size in bytes of shared region */
	uint64_t mapped;				/**< Current map counter  */
	utils_list_s accelerator_list;	/**< List of accelerators */
	utils_list_s process_list;		/**< List of processes */
	utils_queue_s * queue;				/**< Queue */
	arch_alloc_s allocator;		/**< Allocator for this shared memory */
}vine_pipe_s;

/**
 * Return an initialized vine_pipe_s instance.
 *
 * @return An intialized vine_pipe_s instance,NULL on failure.
 */
vine_pipe_s * vine_pipe_get();

/**
 * Initialize a vine_pipe.
 *
 * \note This function must be called from all end points in order to
 * initialize a vine_pipe instance.Concurrent issuers will be serialized
 * and the returned vine_pipe instance will be initialized by the 'first'
 * issuer. All subsequent issuers will receive the already initialized
 * instance.
 *
 * @param mem Shared memory pointer.
 * @param size Size of the shared memory in bytes.
 * @param queue_size Size of all queues in this vine_pipe.
 * @return An initialized vine_pipe_s instance.
 */
vine_pipe_s * vine_pipe_init(void * mem,size_t size,size_t queue_size);

/**
 * Add \c accel in the \c pipe accelerator list.
 *
 * @param pipe The pipe instance where the accelerator is added.
 * @param accel The accelerator to be added.Must be allocated using\
 * 				arch_alloc_alloc(\c pipe,...) and initialized.
 * @return Returns 0 on success.
 */
int vine_pipe_register_accel(vine_pipe_s * pipe,vine_accel_s * accel);

/**
 * Find an accelerator matching the user specified criteria.
 *
 * @param pipe vine_pipe instance.
 * @param name The cstring name of the accelerator, \
 * 				NULL if we dont care for the name.
 * @param type Type of the accelerator, see vine_accel_type_e.
 * @return An vine_accel_s instance, NULL on failure.
 */
vine_accel_s * vine_proc_find_accel(vine_pipe_s * pipe,const char * name,vine_accel_type_e type);

/**
 * Add \c proc in the \c pipe procedure list.
 *
 * @param pipe The pipe instance where the procedure is added.
 * @param proc The procedure to be added.Must be allocated using\
 * 				arch_alloc_alloc(\c pipe,...) and initialized.
 * @return Returns 0 on success.
 */
int vine_pipe_register_proc(vine_pipe_s * pipe,vine_proc_s * proc);

/**
 * Find a procedure matching the user specified criteria.
 *
 * @param pipe vine_pipe instance.
 * @param name The cstring name of the procedure.
 * @param type Type of the procedure, see vine_accel_type_e.
 * @return An vine_proc_s instance, NULL on failure.
 */
vine_proc_s * vine_proc_find_proc(vine_pipe_s * pipe,const char * name,vine_accel_type_e type);

/**
 * Destroy vine_pipe.
 *
 * \note Ensure you perform any cleanup(e.g. delete shared segment)
 * when return value becomes 0.
 *
 * @param pipe vine_pipe instance to be destroyed.
 * @return Number of remaining users of this shared segment.
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