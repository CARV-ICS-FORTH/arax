#ifndef ARAX_DATA_HEADER
#define ARAX_DATA_HEADER
#include <arax.h>
#include "core/arax_object.h"
#include "async.h"
#include <conf.h>
#include "core/arax_accel.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Calculate allocation size for a buffer of size \c SIZE and alignment \c ALIGN.
 */
#define ARAX_BUFF_ALLOC_SIZE(SIZE, ALIGN) ( (SIZE) + (ALIGN) + sizeof(size_t *) )

/**
 * Calculate allocation size for arax_data_s \c DATA.
 */
#define ARAX_DATA_ALLOC_SIZE(DATA)          \
    ARAX_BUFF_ALLOC_SIZE(                   \
        ( arax_data_size(DATA) ),           \
        (((arax_data_s *) (DATA))->align)     \
    )

typedef enum arax_data_flags
{
    OTHR_REMT = 4,
} arax_data_flags_e;

typedef struct arax_data_s arax_data_s;

typedef int (arax_data_sync_fn)(arax_data_s *);

struct arax_data_s
{
    arax_object_s obj; /* Might make this optional (for perf
                        * reasons) */
    void *        remote;
    void *        accel_meta;
    arax_accel *  accel;
    size_t        size;
    size_t        align;
    size_t        flags;
    void *        buffer;
    arax_accel_s *phys;
    #ifdef ARAX_DATA_TRACK
    char *        alloc_track;
    #endif
};

typedef struct arax_data_dtr arax_data_dtr;

struct arax_data_dtr
{
    void * remote;
    size_t size;
    void * phys;
};

/**
 * Initialize a new arax_data_s object.
 * @param vpipe Valid arax_pipe_s instance.
 * @param size Size of data in bytes.
 */
arax_data_s* arax_data_init(arax_pipe_s *vpipe, size_t size);

/**
 * Initialize a new arax_data_s object, with an aligned buffer.
 * @param vpipe Valid arax_pipe_s instance.
 * @param size Size of data in bytes.
 * @param align alignment of buffer in bytes, power of two.
 */
arax_data_s* arax_data_init_aligned(arax_pipe_s *vpipe, size_t size, size_t align);

/**
 * Get the data of \c data, and copy them to \c user.
 *
 * \note This is a blocking call.
 *
 * @param data A valid arax_data_s instance.
 * @param user An allocated memory of at least \c arax_data_size() bytes.
 */
void arax_data_get(arax_data *data, void *user);

/**
 * Set \c data remote (accelerator) buffer to point to \c remt,
 * owned by \c accel.
 * \note This call only be called for arax_data that have no alocated
 * remote buffers (i.e. arax_data_has_remote() returns 0)
 * @param  data Arax data.
 */
void arax_data_set_remote(arax_data_s *data, arax_accel *accel, void *remt);

/**
 * Set accelerator to data and increment reference counters.
 * @param data A valid arax_data_s instance.
 * @param accel Accelerator/fifo to use.
 */
void arax_data_set_accel(arax_data_s *data, arax_accel *accel);

/**
 * Copy data from \c user to 'c data.
 *
 * \note This is a NON blocking call.
 *
 * @param data A valid arax_data_s instance.
 * @param accel Accelerator/fifo to use.
 * @param user An allocated memory of at least \c arax_data_size() bytes.
 */
void arax_data_set(arax_data *data, arax_accel *accel, const void *user);

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
void arax_data_memcpy(arax_accel *accel, arax_data_s *dst, arax_data_s *src, int block);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void arax_data_arg_init(arax_data_s *data, arax_accel *accel);

void arax_data_input_init(arax_data_s *data, arax_accel *accel);

void arax_data_output_init(arax_data_s *data, arax_accel *accel);
#endif

/**
 * Return size of provided arax_data object.
 * @param data Valid arax_data pointer.
 * @return Return size of data of provided arax_data object.
 */
size_t arax_data_size(arax_data *data);

/**
 * Mark \c data for deletion.
 */
void arax_data_free(arax_data *data);

/**
 * Print debug info for 'c data.
 */
void arax_data_stat(arax_data *data, const char *file, size_t line);

#define arax_data_stat(DATA) arax_data_stat(DATA, __FILE__, __LINE__);

#ifdef ARAX_DATA_ANNOTATE
#define arax_data_annotate(DATA, ...)  \
    arax_object_rename((arax_object_s *) DATA, __VA_ARGS__)
#else
#define arax_data_annotate(DATA, ...)
#endif

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef ARAX_DATA_HEADER */
