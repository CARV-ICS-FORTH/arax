#ifndef UTILS_SYSTEM_HEADER
#define UTILS_SYSTEM_HEADER
#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Get current users home directory.
 *
 * \note Do NOT free returned pointer.
 *
 * @return NULL terminated string with home path.
 */
char* system_home_path();

/**
 * Get value of enviroment variable \c var.
 * NULL if variable not set.
 *
 * \note Do NOT free returned pointer.
 *
 * @return NULL terminated string with enviroment variable value.
 */
const char* system_env_var(const char *var);

/**
 * Type to mmap \c file of size \c shm_size.
 *
 * \param base If succesfull will return mmap location, null otherwise. Initial value is used as a hint.
 * \param fd File descriptor. If succesfull will be positive.Negative on failure.
 * \param file Location of mmap backing file.
 * \param shm_size Size of resulting mmap and \c file.
 * \param shm_off Skip bytes from the mmap of \c file.
 * \param truncate Truncate \c file to \c shm_size bytes.
 * \return 0 on success, check \c base and \c fd to figure error.
 */
int system_mmap(void **base, int *fd, const char *file, size_t shm_size, size_t shm_off, int truncate);

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
off_t system_file_size(const char *file);

/**
 * Get current executable name
 */
const char* system_exec_name();

/**
 * Get PID of current process
 */
int system_process_id();

/**
 * Get thread id of current process thread
 */
int system_thread_id();

/**
 * Get stack backtrace for calling thread.
 * Returned string does not end with a new line.
 *
 * \note Do not free or modify returned value
 *
 * \param skip Number of functions to skip from trace.0 will show up to the caller.
 * \return formated acktrace
 */
const char* system_backtrace(unsigned int skip);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */


#endif /* ifndef UTILS_SYSTEM_HEADER */
