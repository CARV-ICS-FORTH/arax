#ifndef VINE_DATA_HEADER
#define VINE_DATA_HEADER
#include <vine_talk.h>
#include "core/vine_object.h"
#include "async.h"
#include "arch/alloc.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef enum vine_data_flags
{
	NONE_SYNC  = 0,
	USER_SYNC = 1,
	SHM_SYNC = 2,
	REMT_SYNC = 4,
	ALL_SYNC  = 7,
	FREE         = 8,
}vine_data_flags_e;

typedef enum vine_data_sync_dir
{
	TO_REMOTE = 1,
	FROM_REMOTE = 2,
}vine_data_sync_dir;

typedef struct vine_data_s vine_data_s;

typedef int (vine_data_sync_fn)(vine_data_s *);

struct vine_data_s {
	vine_object_s obj; /* Might make this optional (for perf
	                    * reasons) */
	vine_pipe_s             *vpipe;
	void                    *user;
	void                    *remote;
	void                    *accel_meta;
	vine_accel              *accel;
	size_t                  size;
    size_t                  align;
	size_t                  flags;
	size_t					sync_dir;
	async_completion_s      ready;
	void                    *buffer;
};

/**
 * Initialize a new vine_data_s object.
 * @param vpipe Valid vine_pipe_s instance.
 * @param user Pointer to user allocated buffer.
 * @param size Size of data in bytes.
 */
vine_data_s* vine_data_init(vine_pipe_s * vpipe,void * user, size_t size);

/**
 * Initialize a data remote pointer aligned.
 * @param  data Vine data.
 * @return Nothing.
 */
vine_object_s* vine_data_allocate(vine_data_s* data);

/**
 * Initialize a new vine_data_s object, with an aligned buffer.
 * @param vpipe Valid vine_pipe_s instance.
 * @param user Pointer to user allocated buffer.
 * @param size Size of data in bytes.
 * @param align alignment of buffer in bytes, power of two.
 */
vine_data_s* vine_data_init_aligned(vine_pipe_s * vpipe,void * user, size_t size,size_t align);

/**
 * Verify data flags are consistent.
 * Will print error message and abort if flags are inconsistent.
 */
void vine_data_check_flags(vine_data_s * data);

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
void vine_data_memcpy(vine_accel * accel,vine_data_s * dst,vine_data_s * src,int block);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void vine_data_arg_init(vine_data_s* data,vine_accel * accel);

void vine_data_input_init(vine_data_s* data,vine_accel * accel);

void vine_data_output_init(vine_data_s* data,vine_accel * accel);

void vine_data_output_done(vine_data_s* data);
#endif

/**
 * Return size of provided vine_data object.
 * @param data Valid vine_data pointer.
 * @return Return size of data of provided vine_data object.
 */
size_t vine_data_size(vine_data *data);

/**
 * Get pointer to buffer for use from CPU.
 *
 * @param data Valid vine_data pointer.
 * @return Ram point to vine_data buffer.NULL on failure.
 */
void* vine_data_deref(vine_data *data);

/**
 * Get pointer to vine_data object from related CPU buffer \c data.
 * Undefined behaviour if \c data is not a value returned by vine_data_deref.
 * @return pointer to vine_data.NULL on failure.
 */
vine_data * vine_data_ref(void * data);

/**
 * Mark data as ready for consumption.
 *
 * @param vpipe Valid vine_pipe_s instance.
 * @param data The vine_data to be marked as ready.
 */
void vine_data_mark_ready(vine_pipe_s *vpipe, vine_data *data);

/**
 * Return if data is marked as ready or not.
 *
 * @param vpipe Valid vine_pipe_s instance.
 * @param data The vine_data to be checked.
 * @return 0 If data is not ready, !0 if data is ready.
 */
int vine_data_check_ready(vine_pipe_s *vpipe, vine_data *data);

/**
 * Mark \c data for deletion.
 */
void vine_data_free(vine_data *data);

/**
 * Send user data to the remote
 *
 * @param accel Accelerator/fifo to use.
 * @param data Data to be synced to remote.
 * @param block If !=0 this call will block until data are synced to remote.
 */
void vine_data_sync_to_remote(vine_accel * accel,vine_data * data,int block);

/**
 * Get remote data to user
 *
 * @param accel Accelerator/fifo to use.
 * @param data Data to be synced from remote.
 * @param block If !=0 this call will block until data are synced to remote.
 */
void vine_data_sync_from_remote(vine_accel * accel,vine_data * data,int block);

/*
 * Mark where \c data is modified.
 *
 * Will invalidate all other regions.
 */
void vine_data_modified(vine_data * data,vine_data_flags_e where);

/**
 * Print debug info for 'c data.
 */
void vine_data_stat(vine_data * data,const char * file,size_t line);

#define vine_data_stat(DATA) vine_data_stat(DATA,__FILE__,__LINE__);

#ifdef VINE_DATA_ANNOTATE
	#define vine_data_annotate(DATA, ...)  \
		vine_object_rename((vine_object_s*)DATA,__VA_ARGS__)
#else
	#define vine_data_annotate(DATA, ...)
#endif

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_DATA_HEADER */
