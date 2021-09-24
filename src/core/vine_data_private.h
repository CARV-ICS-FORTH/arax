#ifndef VINE_DATA_PRIVATE_HEADER
#define VINE_DATA_PRIVATE_HEADER
#include "vine_data.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Transfer data between shm and remote.
 *
 * @param accel Accelerator/fifo to use.
 * @param func Sync function to use. Can be "syncTo" or "syncFrom"
 * @param data Data to be moved with \c func.
 * @param block If !=0 this call will block until data are moved.
 */
void vine_data_shm_sync(vine_accel *accel, const char *func, vine_data_s *data, int block);

/**
 * Migrate \c data accelerator location to \c accel.
 *
 * \NOTE: Does not yet support migration across physical devices.
 */
void vine_data_migrate_accel(vine_data_s *data, vine_accel *accel);

/**
 * Initialize \c data remote (accelerator) buffer.
 * @param  data Vine data.
 */
void vine_data_allocate_remote(vine_data_s *data, vine_accel *accel);

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
vine_data* vine_data_ref(void *data);

/**
 * Get pointer to vine_data object from \c data that points 'inside' related CPU buffer .
 * @return pointer to vine_data.NULL on failure.
 */
vine_data* vine_data_ref_offset(vine_pipe_s *vpipe, void *data);

/**
 * Returns true if \c data has been allocated on the remote accelerator.
 *
 * @param data Data to be queried.
 * @return 1 if \c data has a remote accelerator allocation, 0 otherwise.
 */
int vine_data_has_remote(vine_data *data);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef VINE_DATA_PRIVATE_HEADER
