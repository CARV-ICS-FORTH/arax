#ifndef ARAXCONTROLLER_HIP_UTILS
#define ARAXCONTROLLER_HIP_UTILS
#include "definesEnable.h"
#include "hip/hip_runtime.h"
#include "utils/arax_assert.h"
#include <iostream>

#define RED   "\033[1;31m"
#define RESET "\033[0m"

#ifdef ERROR_CHECKING
#define HIP_ERROR_FATAL(err)                                                   \
    hipErrorCheckFatal(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused)) hipErrorCheckFatal(hipError_t err, const char *func, const char *file,
  size_t line)
{
    if (err != hipSuccess) {
        std::cerr << RED << func << " error : " << RESET << hipGetErrorString(err)
                  << std::endl;
        std::cerr << "\t" << file << RED << " Failed at " << RESET << line
                  << std::endl;
        arax_assert(!"Fatality");
    }
}

#else // ifdef ERROR_CHECKING
#define HIP_ERROR_FATAL(err)
#endif // ifdef ERROR_CHECKING
#endif // ifndef ARAXCONTROLLER_HIP_UTILS
