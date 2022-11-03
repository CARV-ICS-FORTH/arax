#include <cublas_v2.h>
#include <iostream>

#include "Services.h"

class CublasService : public Service {
public:
  bool supports(accelThread *thread) const {
    return thread->getAccelType() == GPU;
  }
  static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
  }
  void *createInstance(accelThread *thread) {
    cublasStatus_t stat;
    cublasHandle_t instance = 0;
    stat = cublasCreate(&instance);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << __func__ << " cublas failed init faied with "
                << _cudaGetErrorEnum(stat) << std::endl;
      throw std::runtime_error("Could not create Cublas instance!\n");
    }

    return instance;
  }

  void destroyInstance(accelThread *thread, void *instance) {

    if (!instance)
      throw std::runtime_error(
          "Attempting to destroy invalid Cublas instance!\n");
    if (cublasDestroy((cublasHandle_t)instance) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Could not destroy Cublas instance!\n");
    }
  }
};

REGISTER_SERVICE(CublasService)
