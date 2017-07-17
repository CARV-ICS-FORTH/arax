#ifndef VINE_DATA_HEADER
#define VINE_DATA_HEADER
#include <vine_talk.h>
#include "core/vine_object.h"
#include "async.h"
#include "arch/alloc.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * vine_data: Opaque data pointer.
 */
typedef void vine_data;

/**
 * Allocation strategy enumeration.
 */
typedef enum vine_data_alloc_place {
	HostOnly  = 1, /**< Allocate space only on host memory(RAM) */
	AccelOnly = 2, /**< Allocate space only on accelerator memory (e.g. GPU
	* VRAM) */
	Both      = 3 /**< Allocate space on both host memory and accelerator
	* memory. */
} vine_data_alloc_place_e;

typedef struct vine_data_s {
	vine_object_s obj; /* Might make this optional (for perf
	                    * reasons) */
	vine_data_alloc_place_e place;
	size_t                  size;
	size_t                  flags;
	async_completion_s ready;

	/* Add status variables */
} vine_data_s;
typedef enum vine_data_io {
	VINE_INPUT = 1, VINE_OUTPUT = 2
} vine_data_io_e;

vine_data_s* vine_data_init(vine_pipe_s * vpipe, size_t size,
                            vine_data_alloc_place_e place);


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

void vine_data_free(vine_pipe_s *vpipe, vine_data *data);

void vine_data_erase(vine_object_repo_s *repo, vine_data_s *data);

int vine_data_valid(vine_object_repo_s *repo, vine_data_s *data);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_DATA_HEADER */
