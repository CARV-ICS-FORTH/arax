#ifndef UTILS_COMPAT_HEADER
#define UTILS_COMPAT_HEADER

/**
 * Placeholder struct to be used instead of an empty struct.
 *
 * This struct is necessary as a c struct is 0 bytes*, whereas a
 * c++ struct is 1 byte big**, resulting in problematic c<->c++ interaction.
 *
 * https://gcc.gnu.org/onlinedocs/gcc/Empty-Structures.html#Empty-Structures
 * ** Could not find reference, but seems to be defined as 'non zero'.
 */
typedef struct
{
    unsigned long nothing;
} utils_compat_empty_s;

/**
 * Macro to allow the use of C linked functions from C++ code/programs.
 */
#ifdef __cplusplus
#define ARAX_CPP extern "C"
#else
#define ARAX_CPP
#endif

#endif // ifndef UTILS_COMPAT_HEADER
