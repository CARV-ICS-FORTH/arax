#ifndef ARAX_TALK_TYPES_HEADER
#define ARAX_TALK_TYPES_HEADER
#include <sys/time.h>
#include <time.h>

/**
 * arax_accel: Accelerator descriptor.
 */
typedef void arax_accel;

/**
 * arax_proc: Process descriptor.
 */
typedef void arax_proc;

/**
 * Arax Pipe instance
 */
typedef struct arax_pipe arax_pipe_s;

/**
 * Accelerator Statistics
 */
typedef struct arax_accel_stats
{
    // This padp is necessary as empty struct have sizeof == 1 in C++, but 0 in C
    #ifndef __cplusplus
    char padd;
    #endif
} arax_accel_stats_s;

typedef struct utils_timer_s
{
    struct timespec start;
    struct timespec stop;
} utils_timer_s;

/**
 * Accelerator State enumeration.
 */
typedef enum arax_accel_state
{
    accel_failed, /**< Accelerator has failed. */
    accel_idle,   /**< Accelerator is idle. */
    accel_busy    /**< Accelerator is busy. */
} arax_accel_state_e;

/**
 * Arax Task Descriptor
 */
typedef void arax_task;

/**
 * arax_data: Opaque data pointer.
 */
typedef void arax_data;

/**
 * Arax Task State enumeration.
 */
typedef enum arax_task_state_e
{
    task_failed,   /**< Task execution failed. */
    task_issued,   /**< Task has been issued. */
    task_completed /**< Task has been completed. */
} arax_task_state_e;

/**
 * Arax Task Statistics
 */
typedef struct arax_task_stats
{
    int           usedSlots;
    utils_timer_s task_duration_without_issue;
    utils_timer_s task_duration;
} arax_task_stats_s;

/**
 * Accelerator type enumeration.
 * NOTE: If updated update types_map variable in arax_accel_types.c
 */
typedef enum arax_accel_type
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
    ARAX_ACCEL_TYPES /** End Marker                            */
} arax_accel_type_e;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct arch_alloc_s arch_alloc_s;

typedef struct async_meta_s async_meta_s;

typedef void *arax_buffer_s;

/**
 * Receives arguments and inputs/outputs.
 * Performs argument marshalling and task issue to accelerator.
 */
typedef arax_task_state_e (AraxFunctor)(arax_task *);

#endif // ifndef ARAX_TALK_TYPES_HEADER
