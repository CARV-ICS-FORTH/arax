#ifndef ARAX_DATA_PRIVATE_HEADER
#define ARAX_DATA_PRIVATE_HEADER
#include "arax_data.h"

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
void arax_data_shm_sync(arax_accel *accel, const char *func, arax_data_s *data, int block);

/**
 * Migrate \c data accelerator location to \c accel.
 *
 * \NOTE: Does not yet support migration across physical devices.
 */
void arax_data_migrate_accel(arax_data_s *data, arax_accel *accel);

/**
 * Initialize \c data remote (accelerator) buffer.
 * @param  data Arax data.
 */
void arax_data_allocate_remote(arax_data_s *data, arax_accel *accel);

/**
 * Get pointer to buffer for use from CPU.
 *
 * @param data Valid arax_data pointer.
 * @return Ram point to arax_data buffer.NULL on failure.
 */
void* arax_data_deref(arax_data *data);

/**
 * Get pointer to arax_data object from related CPU buffer \c data.
 * Undefined behaviour if \c data is not a value returned by arax_data_deref.
 * @return pointer to arax_data.NULL on failure.
 */
arax_data* arax_data_ref(void *data);

/**
 * Get pointer to arax_data object from \c data that points 'inside' related CPU buffer .
 * @return pointer to arax_data.NULL on failure.
 */
arax_data* arax_data_ref_offset(arax_pipe_s *vpipe, void *data);

/**
 * Returns true if \c data has been allocated on the remote accelerator.
 *
 * @param data Data to be queried.
 * @return 1 if \c data has a remote accelerator allocation, 0 otherwise.
 */
int arax_data_has_remote(arax_data *data);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef ARAX_DATA_PRIVATE_HEADER
