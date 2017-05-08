#ifndef VINE_TALK_TYPES_HEADER
#define VINE_TALK_TYPES_HEADER
#include <sys/time.h>
#include <time.h>
/**
 * vine_accel: Accelerator descriptor.
 */
typedef void vine_accel;

/**
 * vine_proc: Process descriptor.
 */
typedef void vine_proc;

/**
 * Location of a vine_accel.
 */
typedef struct vine_accel_loc {
	/**< To be filled */
} vine_accel_loc_s;

/**
 * Vine Pipe instance
 */
typedef struct vine_pipe vine_pipe_s;

/**
 * Accelerator Statistics
 */
typedef struct vine_accel_stats {} vine_accel_stats_s;

typedef struct utils_timer_s
{
	struct timespec start;
	struct timespec stop;
}utils_timer_s;

/**
 * Accelerator State enumeration.
 */
typedef enum vine_accel_state {
	accel_failed, /**< Accelerator has failed. */
	accel_idle, /**< Accelerator is idle. */
	accel_busy /**< Accelerator is busy. */
} vine_accel_state_e;

/**
 * Vineyard Task Descriptor
 */
typedef void vine_task;

/**
 * Vine Task State enumeration.
 */
typedef enum vine_task_state_e {
	task_failed, /**< Task execution failed. */
	task_issued, /**< Task has been issued. */
	task_completed /**< Task has been completed. */
} vine_task_state_e;

/**
 * Vine Task Statistics
 */
typedef struct vine_task_stats {
	int task_id; /**< Unique among tasks of this instance */
	utils_timer_s task_duration;
} vine_task_stats_s;

/**
 * Accelerator type enumeration.
 * NOTE: If updated update types_map variable in vine_accel_types.c
 */
typedef enum vine_accel_type {
	ANY       = 0,   /**< Let Scheduler Decide                 */
	GPU       = 1,   /**< Run on GPU with CUDA                 */
	GPU_SOFT  = 2,   /**< Run on CPU with software CUDA        */
	CPU       = 3,   /**< Run Native x86 code                  */
	SDA       = 4,   /**< Xilinx SDAaccel                      */
	NANO_ARM  = 5,   /**< ARM accelerator core from NanoStream */
	NANO_CORE = 6,   /**< NanoStreams FPGA accelerator         */
	VINE_ACCEL_TYPES /** End Marker                            */
} vine_accel_type_e;

/**
 * Convert a vine_accel_type_e value to a human readable string.
 * If \c type not a valid vine_accel_type_e value NULL is returned.
 * NOTE: This function should not be used in critical paths!
 *
 * @return A cahracter representation for the given \c type,NULL on error.
 */
const char * vine_accel_type_to_str(vine_accel_type_e type);

/**
 * Convert a string to the matching vine_accel_type_e value.
 * \c type will be compared ignoring capitalization with the string in
 * types_map variable in vine_accel_types.c.
 *
 * NOTE: This function should not be used in critical paths!
 *
 * @return A value from vine_accel_type_e, if no match is found returns
 * VINE_ACCEL_TYPES
 */
vine_accel_type_e vine_accel_type_from_str(const char * type);

typedef struct arch_alloc_s arch_alloc_s;

typedef struct vine_buffer_s vine_buffer_s;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct async_meta_s async_meta_s;
#endif
