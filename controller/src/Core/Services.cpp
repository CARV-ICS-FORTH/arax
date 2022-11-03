#include "Services.h"
#include "Utilities.h"
#include <iostream>

#include <sys/time.h>

ServiceProvider serviceProvider;

void * Service ::createNewInstance(accelThread *thread)
{
    return instances[thread] = createInstance(thread);
}

void * ServiceProvider ::createService(accelThread *thread,
  const char *                                      service)
{
    Service *s = 0;

    // create the service provider
    s = constructType(service);
    // add to the map of vectors the new service
    if (!s->supports(thread))
        return 0;

    // create the actual hanlder/generator
    void *instance = s->createNewInstance(thread);

    activeServicesPerThread[instance] = s;
    // std::cerr<<__func__<<" ,service: "<<service<<" ,handle/generator:
    // "<<instance<<std::endl;
    return instance;
}

void ServiceProvider ::destroyService(accelThread *thread, void *srv2Stop)
{
    // std::cerr<<__func__<<" ,service: "<<service<<" ,service2Stop:
    // "<<srv2Stop<<std::endl;
    Service *srv = activeServicesPerThread.at(srv2Stop);

    if (srv->supports(thread)) {
        srv->destroyInstance(thread, srv2Stop);
    }
    activeServicesPerThread.erase(srv2Stop);
}
