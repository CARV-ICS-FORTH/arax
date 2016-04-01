#ifndef VINE_PROC_HEADER
#define VINE_PROC_HEADER
#include "vine_talk.h"
#include "structs/list.h"

typedef struct {
	structs_list_node_s  list;
	vine_accel_type_e type;
	int               users;
	int               data_off; /**< Offset relative to name where process
	                             * binary begins(strlen(name)) */
	char              name[];

	/* To add more as needed */
} vine_proc_s;

/**
 * Initialize a vine_proc at the memory pointed by \c mem.
 *
 * @param mem Allocated memory of size > vine_proc_calc_size().
 * @param name NULL terminated string, will be copied to provate buffer.
 * @param type Accelerator type.
 * @param code Pointer to bytes containing procedure executable.
 * @param code_size Size of \c code parameter
 * @return An initialized instance of vine_proc_s, NULL on failure.
 */
vine_proc_s* vine_proc_init(void *mem, const char *name, vine_accel_type_e type,
                            const void *code, size_t code_size);

size_t vine_proc_calc_size(const char *name, size_t code_size);

#endif /* ifndef VINE_PROC_HEADER */
