#ifndef VINE_ACCEL_HEADER
#define VINE_ACCEL_HEADER
#include <vine_talk.h>
#include "async.h"
typedef struct vine_accel_s vine_accel_s;

#include "core/vine_vaccel.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

struct vine_accel_s {
	vine_object_s      obj;
	vine_accel_type_e  type;
	utils_spinlock     lock;
	utils_list_s       vaccels;
	vine_accel_loc_s   location;
	vine_accel_stats_s stats;
	vine_accel_state_e state;
	size_t             revision;
    size_t        AvaliableSize;
    async_condition_s  gpu_ready;
	/* To add more as needed */
};

/**
 * Allocate and initialize a vine_accel descriptor with the provided arguements.
 * @pipe A valid vine_pipe_s* instance.
 * @pipe Name of new accelerator.
 * @type Accelerator type/architecture.
 * @size Avaliable accelerator memory in bytes.
 * @return An initialized vine_accel instance on success, or NULL on failure.
 */
vine_accel_s* vine_accel_init(vine_pipe_s * pipe, const char *name,
                              vine_accel_type_e type,size_t size);

/**
 * Get name.
 */
const char* vine_accel_get_name(vine_accel_s *accel);

/**
 * Get stats.
 * 
 * @param accel A physsical accelerator
 */
vine_accel_state_e vine_accel_get_stat(vine_accel_s *accel,vine_accel_stats_s * stat);

/**
 * Increase 'revision' of accelerator.
 * 
 * @param accel A physsical accelerator
 */
void vine_accel_inc_revision(vine_accel_s * accel);

/**
 * Get 'revision' of accelerator.
 * 
 * @param accel A physsical accelerator
 * @return 		Revision
 */
size_t vine_accel_get_revision(vine_accel_s * accel);

/**
 * Increments avaliable size of gpu by sz
 *
 * @param vaccel Virtual accelator to set physical accelerator
 * @param sz     Size of added data
 * @return       Nothing .
 */
void vine_accel_size_inc(vine_accel* vaccel,size_t sz);

/**
 * Decrements avaliable size of gpu by sz
 *
 * @param vaccel virtual accelator to set physical accelerator
 * @param sz     size of removed data
 * @return       Nothing .
 */
void vine_accel_size_dec(vine_accel* vaccel,size_t sz);

/**
 * Gets avaliable size of GPU
 *
 * @param vaccel Virtual accelator to set physical accelerator
 * @return       Avaliable size of GPU 
 */
size_t vine_accel_get_size(vine_accel* vaccel);

/**
 * Gets avaliable size of GPU
 *
 * @param accel  Physical accelator to set physical accelerator
 * @return       Avaliable size of GPU 
 */
size_t vine_accel_get_AvaliableSize(vine_accel_s* accel);

/**
 * Add (register) a virtual accell \c vaccel to physical accelerator \c accel.
 *
 * @param accel A physsical accelerator
 * @param vaccel A virtual accelerator to be linked with \c accel
 */
void vine_accel_add_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel);

/**
 * Delete (unregister) a virtual accell \c vaccel from physical accelerator \c accel.
 *
 * @param accel A physsical accelerator
 * @param vaccel A virtual accelerator to be unlinked from \c accel
 */
void vine_accel_del_vaccel(vine_accel_s * accel,vine_vaccel_s * vaccel);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_ACCEL_HEADER */
