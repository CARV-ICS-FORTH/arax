#include "Services.h"
#include <curand.h>
#include <iostream>

class CurandService : public Service {
public:
  bool supports(accelThread *thread) const {
    return thread->getAccelType() == GPU;
  }
  void *createInstance(accelThread *thread) {
    curandGenerator_t instance = 0;

    if (curandCreateGenerator(&instance, CURAND_RNG_PSEUDO_DEFAULT) !=
        CURAND_STATUS_SUCCESS) {
      std::cerr << __func__ << ": " << __FILE__ << " curand created FAILED!!!!!"
                << std::endl;
      throw std::runtime_error("Could not create curand instance!\n");
    }

    // std::cerr << "Curand is go @" << (void*)thread << " : " <<
    // (void*)instance << std::endl;
    return instance;
  }
  void destroyInstance(accelThread *thread, void *instance) {
    if (!instance)
      throw std::runtime_error(
          "Attempting to destroy invalid curand instace!\n");

    if (curandDestroyGenerator((curandGenerator_t)instance) !=
        CURAND_STATUS_SUCCESS) {
      std::cerr << __func__ << ": " << __FILE__ << " curand destroy FAILED!!!!!"
                << std::endl;
      throw std::runtime_error("Could not destroy curand instance!\n");
    }
    // std::cerr << "Curand is dead @" << (void*)thread << " : " <<
    // (void*)instance << std::endl;
  }
};

REGISTER_SERVICE(CurandService)
