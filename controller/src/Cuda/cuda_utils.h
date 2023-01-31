#ifndef ARAXCONTROLLER_CUDA_UTILS
#define ARAXCONTROLLER_CUDA_UTILS
#include "definesEnable.h"
#include "utils/arax_assert.h"
#include <iostream>
#define RED   "\033[1;31m"
#define GREEN "\033[1;32m"
#define RESET "\033[0m"
#ifdef ERROR_CHECKING
#define CUDA_ERROR_FATAL(err)                                                  \
    cudaErrorCheckFatal(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused)) cudaErrorCheckFatal(cudaError_t err, const char *func, const char *file,
  size_t line)
{
    if (err != cudaSuccess) {
        std::cerr << RED << func << " error : " << RESET << cudaGetErrorString(err)
                  << std::endl;
        std::cerr << "\t" << file << RED << " Failed at " << RESET << line
                  << std::endl;
        arax_assert(!"Fatality");
    }
}

#else // ifdef ERROR_CHECKING
#define CUDA_ERROR_FATAL(err) ((void) err)
#endif // ifdef ERROR_CHECKING
#endif // ifndef ARAXCONTROLLER_CUDA_UTILS
