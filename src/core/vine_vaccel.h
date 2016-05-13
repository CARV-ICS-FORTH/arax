#ifndef VINE_VACCEL_HEADER
#define VINE_VACCEL_HEADER
#include <vine_talk.h>
#include "utils/queue.h"
#include "core/vine_object.h"
#include "core/vine_accel.h"

/**
 * Virtual Accelerator
 *
 * Creates a dedicated queue mapped to a physical accelerator.
 */
typedef struct {
	vine_object_s  obj;
	utils_spinlock lock;
	vine_accel_s   *phys;
} vine_vaccel_s;

/**
 * Initialize a vine_vaccel_s in \c mem.
 *
 * \param repo A valid vine_object_repo_s instance
 * \param mem An allocated memory buffer
 * \param mem_size The size of the \c mem buffer in bytes
 * \param name Name of the virtual accelerator
 * \param accel A physical accelerator
 */
vine_vaccel_s* vine_vaccel_init(vine_object_repo_s *repo, void *mem,
                                size_t mem_size, char *name,
                                vine_accel_s *accel);

/**
 * Get the queue of \c vaccel.
 *
 * \param vaccel A virtual accelerator
 * \return The queue of \c vaccel
 */
utils_queue_s* vine_vaccel_queue(vine_vaccel_s *vaccel);

/**
 * Erase \c accel from the list of virtual accelerators.
 *
 * \param repo A valid vine_object_repo_s instance
 * \param vaccel The virtual accelerator to be erased
 */
void vine_vaccel_erase(vine_object_repo_s *repo, vine_vaccel_s *accel);

#endif /* ifndef VINE_VACCEL_HEADER */
