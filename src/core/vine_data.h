#ifndef VINE_DATA_HEADER
#define VINE_DATA_HEADER
#include <vine_talk.h>
#include "core/vine_object.h"
#include "async.h"
#include <conf.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Calculate allocation size for a buffer of size \c SIZE and alignment \c ALIGN.
 */
#define VINE_BUFF_ALLOC_SIZE(SIZE, ALIGN) ( (SIZE) + (ALIGN) + sizeof(size_t *) )

/**
 * Calculate allocation size for vine_data_s \c DATA.
 */
#define VINE_DATA_ALLOC_SIZE(DATA)          \
    VINE_BUFF_ALLOC_SIZE(                   \
        ( vine_data_size(DATA) ),           \
        (((vine_data_s *) (DATA))->align)     \
    )

typedef enum vine_data_flags
{
    NONE_SYNC = 0,
    SHM_SYNC  = 1,
    REMT_SYNC = 2,
    ALL_SYNC  = 3,
    OTHR_REMT = 4,
} vine_data_flags_e;

typedef struct vine_data_s vine_data_s;

typedef int (vine_data_sync_fn)(vine_data_s *);

struct vine_data_s
{
    vine_object_s obj; /* Might make this optional (for perf
                        * reasons) */
    void *        remote;
    void *        accel_meta;
    vine_accel *  accel;
    size_t        size;
    size_t        align;
    size_t        flags;
    void *        buffer;
    #ifdef VINE_DATA_TRACK
    char *        alloc_track;
    #endif
};

typedef struct vine_data_dtr vine_data_dtr;

struct vine_data_dtr
{
    void * remote;
    size_t size;
    void * phys;
};

/**
 * Initialize a new vine_data_s object.
 * @param vpipe Valid vine_pipe_s instance.
 * @param size Size of data in bytes.
 */
vine_data_s* vine_data_init(vine_pipe_s *vpipe, size_t size);

/**
 * Initialize a new vine_data_s object, with an aligned buffer.
 * @param vpipe Valid vine_pipe_s instance.
 * @param size Size of data in bytes.
 * @param align alignment of buffer in bytes, power of two.
 */
vine_data_s* vine_data_init_aligned(vine_pipe_s *vpipe, size_t size, size_t align);

/**
 * Get the data of \c data, and copy them to \c user.
 *
 * \note This is a blocking call.
 *
 * @param data A valid vine_data_s instance.
 * @param user An allocated memory of at least \c vine_data_size() bytes.
 */
void vine_data_get(vine_data *data, void *user);

/**
 * Set \c data remote (accelerator) buffer to point to \c remt,
 * owned by \c accel.
 * \note This call only be called for vine_data that have no alocated
 * remote buffers (i.e. vine_data_has_remote() returns 0)
 * @param  data Vine data.
 */
void vine_data_set_remote(vine_data_s *data, vine_accel *accel, void *remt);

/**
 * Set accelerator to data and increment reference counters.
 * @param data A valid vine_data_s instance.
 * @param accel Accelerator/fifo to use.
 */
void vine_data_set_accel(vine_data_s *data, vine_accel *accel);

/**
 * Copy data from \c user to 'c data.
 *
 * \note This is a NON blocking call.
 *
 * @param data A valid vine_data_s instance.
 * @param accel Accelerator/fifo to use.
 * @param user An allocated memory of at least \c vine_data_size() bytes.
 */
void vine_data_set(vine_data *data, vine_accel *accel, const void *user);

/**
 * Verify data flags are consistent.
 * Will print error message and abort if flags are inconsistent.
 */
void vine_data_check_flags(vine_data_s *data);

/**
 * Copy data of \c src to \c dst.
 *
 * @Note \c src and \c dst must have the same size.
 * @Note If \c src and \c dst are the same, function is no-op.
 *
 * @param accel Accelerator/fifo to use.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param block If true function returns only when copy has completed.
 */
void vine_data_memcpy(vine_accel *accel, vine_data_s *dst, vine_data_s *src, int block);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void vine_data_arg_init(vine_data_s *data, vine_accel *accel);

void vine_data_input_init(vine_data_s *data, vine_accel *accel);

void vine_data_output_init(vine_data_s *data, vine_accel *accel);
#endif

/**
 * Return size of provided vine_data object.
 * @param data Valid vine_data pointer.
 * @return Return size of data of provided vine_data object.
 */
size_t vine_data_size(vine_data *data);

/**
 * Mark \c data for deletion.
 */
void vine_data_free(vine_data *data);

/**
 * Print debug info for 'c data.
 */
void vine_data_stat(vine_data *data, const char *file, size_t line);

#define vine_data_stat(DATA) vine_data_stat(DATA, __FILE__, __LINE__);

#ifdef VINE_DATA_ANNOTATE
#define vine_data_annotate(DATA, ...)  \
    vine_object_rename((vine_object_s *) DATA, __VA_ARGS__)
#else
#define vine_data_annotate(DATA, ...)
#endif

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_DATA_HEADER */
