#ifndef UTILS_ARAX_PLOT_HEADER
#define UTILS_ARAX_PLOT_HEADER
#include <stdint.h>
#include <arax.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Register a new metric \c name and set/get the location where its value resides.
 *
 * The value pointer \c metric can be null or a valid arax pointer.
 * When null, the call will allocate space for the metric, and return the pointer.
 * When given a pointer where \c arax_ptr_valid()==true, the provided pointer is used and returned.
 *
 * \note This should be called before starting the arax_plot executable.
 *
 * \param name Short name of new metric.
 * \param metric NULL or pointer that has \c arax_ptr_valid()==true.
 * \return Pointer location of metric.
 */
uint64_t* arax_plot_register_metric(const char *name, uint64_t *metric);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef UTILS_ARAX_PLOT_HEADER
