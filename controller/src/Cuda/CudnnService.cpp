#include "Core/Services.h"
#include "cudnn.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

class CudnnService : public Service {
public:
    bool supports(accelThread *thread) const
    {
        return thread->getAccelType() == GPU;
    }

    void* createInstance(accelThread *thread)
    {
        cudnnHandle_t instance = 0;

        if (cudnnCreate(&instance) != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error("Could not create cudnn instance!\n");

        // std::cerr << "Cudnn is go @" << (void*)thread << " : " << (void*)instance
        // << std::endl;
        return instance;
    }

    void destroyInstance(accelThread *thread, void *instance)
    {
        if (!instance) {
            throw std::runtime_error(
                      "Attempting to destroy invalid curand instace!\n");
        }

        cudnnDestroy((cudnnHandle_t) instance);
        // std::cerr << "Cudnn is dead @" << (void*)thread << " : " <<
        // (void*)instance << std::endl;
    }
};

REGISTER_SERVICE(CudnnService)
