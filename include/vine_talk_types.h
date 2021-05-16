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
typedef struct vine_accel_loc
{
    /**< To be filled */
} vine_accel_loc_s;

/**
 * Vine Pipe instance
 */
typedef struct vine_pipe vine_pipe_s;

/**
 * Accelerator Statistics
 */
typedef struct vine_accel_stats { } vine_accel_stats_s;

typedef struct utils_timer_s
{
    struct timespec start;
    struct timespec stop;
} utils_timer_s;

/**
 * Accelerator State enumeration.
 */
typedef enum vine_accel_state
{
    accel_failed, /**< Accelerator has failed. */
    accel_idle,   /**< Accelerator is idle. */
    accel_busy    /**< Accelerator is busy. */
} vine_accel_state_e;

/**
 * Vineyard Task Descriptor
 */
typedef void vine_task;

/**
 * vine_data: Opaque data pointer.
 */
typedef void vine_data;

/**
 * Vine Task State enumeration.
 */
typedef enum vine_task_state_e
{
    task_failed,   /**< Task execution failed. */
    task_issued,   /**< Task has been issued. */
    task_completed /**< Task has been completed. */
} vine_task_state_e;

/**
 * Vine Task Statistics
 */
typedef struct vine_task_stats
{
    int           task_id; /**< Unique among tasks of this instance */
    int           usedSlots;
    utils_timer_s task_duration_without_issue;
    utils_timer_s task_duration;
} vine_task_stats_s;

/**
 * Accelerator type enumeration.
 * NOTE: If updated update types_map variable in vine_accel_types.c
 */
typedef enum vine_accel_type
{
    ANY       = 0,   /**< Let Scheduler Decide                 */
    GPU       = 1,   /**< Run on GPU with CUDA                 */
    GPU_SOFT  = 2,   /**< Run on CPU with software CUDA        */
    CPU       = 3,   /**< Run Native x86 code                  */
    SDA       = 4,   /**< Xilinx SDAaccel                      */
    NANO_ARM  = 5,   /**< ARM accelerator core from NanoStream */
    NANO_CORE = 6,   /**< NanoStreams FPGA accelerator         */
    OPEN_CL   = 7,   /**< OpenCl Accelerators                  */
    HIP       = 8,   /**< AMD                                  */
    VINE_ACCEL_TYPES /** End Marker                            */
} vine_accel_type_e;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct async_meta_s async_meta_s;

typedef void *vine_buffer_s;

/**
 * Receives arguments and inputs/outputs.
 * Performs argument marshalling and task issue to accelerator.
 */
typedef vine_task_state_e (VineFunctor)(vine_task *);

#endif // ifndef VINE_TALK_TYPES_HEADER
