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
	USER_IN_SYNC = 1,
	REMT_IN_SYNC = 2,
	ALL_IN_SYNC = 3,
	FREE         = 4
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
	vine_data_sync_fn       *to_remote;
	vine_data_sync_fn       *from_remote;
	vine_data_sync_fn       *free_remote;
	size_t                  size;
	size_t                  flags;
	size_t					sync_dir;
	async_completion_s ready;

	/* Add status variables */
};

vine_data_s* vine_data_init(vine_pipe_s * vpipe,void * user, size_t size);

void vine_data_input_init(vine_data_s* data);

void vine_data_output_init(vine_data_s* data);

void vine_data_set_sync_ops(vine_data_s* data,void *accel_meta,vine_data_sync_fn *to_remote,vine_data_sync_fn *from_remote,vine_data_sync_fn *free_remote);

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
 * Mark data as ready for consumption.
 *
 * @param data The vine_data to be marked as ready.
 */
void vine_data_mark_ready(vine_pipe_s *vpipe, vine_data *data);

/**
 * Return if data is marked as ready or not.
 *
 * @param data The vine_data to be checked.
 * @return 0 If data is not ready, !0 if data is ready.
 */
int vine_data_check_ready(vine_pipe_s *vpipe, vine_data *data);

void vine_data_free(vine_data *data);

int vine_data_valid(vine_object_repo_s *repo, vine_data *data);

/*
 * Send user data to the remote
 */
void vine_data_sync_to_remote(vine_data * data,vine_data_flags_e upto);

/*
 * Get remote data to user
 */
void vine_data_sync_from_remote(vine_data * data,vine_data_flags_e upto);

/*
 * data modified.
 */
void vine_data_modified(vine_data * data,vine_data_flags_e where);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_DATA_HEADER */
