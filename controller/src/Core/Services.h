#ifndef SERVICES_HEADER
#define SERVICES_HEADER
#include "Factory.h"
#include "accelThread.h"
#include <unordered_map>

/*
 * Abstract class allowing the initialization of accelerator specific services.
 */
class Service {
public:
    virtual bool supports(accelThread *thread) const = 0;
    virtual void destroyInstance(accelThread *thread, void *instance) = 0;
    void* createNewInstance(accelThread *thread);

private:
    virtual void* createInstance(accelThread *thread) = 0;
    std::unordered_map<accelThread *, void *> instances;
};

class ServiceProvider : public Factory<Service> {
public:

    /**
     * Create a service (ie. handle or generator) for the specified \c thread
     * accelerator. One app can have mulitple services as such each accelerator
     * thread can manage multiple services.
     **/
    void* createService(accelThread *thread, const char *service);

    /**
     * Release the service (ie. handle, generator), spedified
     * from the application, for the specified \c thread accelerator.
     **/
    void destroyService(accelThread *thread, void *srv2Stop);

private:

    /** Multiple Services that have been instantiated.
     * activeServicesPerThread is Map with key the handle/generator
     * and value the Service. So each thread can
     * have multiple active services. **/
    std::map<void *, Service *> activeServicesPerThread;
};

extern ServiceProvider serviceProvider;

#define REGISTER_SERVICE(CLASS)                                                \
    static Registrator<Service, CLASS> reg(serviceProvider);

/**
 * Declare that \c CLASS requires service \c REQUIRED to operate.
 */
#define REQUIRE_SERVICE(CLASS, REQUIRED)

#endif // ifndef SERVICES_HEADER
