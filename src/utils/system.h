#ifndef UTILS_SYSTEM_HEADER
#define UTILS_SYSTEM_HEADER
#include <stddef.h>
#include <sys/types.h>
/**
 * Get current users home directory.
 *
 * \note Do NOT free returned pointer.
 *
 * @return NULL terminated string with home path.
 */
char* system_home_path();

/**
 * Return total memory in bytes.
 *
 * @return Total memory in bytes.
 */
size_t system_total_memory();

/**
 * Get size of \c file in bytes.
 *
 * @return File size in bytes, 0 on failure.
 */
off_t system_file_size(const char * file);

#endif /* ifndef UTILS_SYSTEM_HEADER */
