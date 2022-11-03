

#include <cuda.h>
#include <cupti.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <occupancy.cuh>

//#define SYNCHRONIZE_WITH_EVENT

CUpti_EventID idWarps;
CUpti_EventID idCycles;

int cuptiEventSet = 0;
cupti_eventData *cuptiEvent;
CUpti_SubscriberHandle subscriber;

cudaEvent_t start_event, stop_event;

int testComplete;

#define CHECK_CU_ERROR(err, cufunc)                                     \
    if (err != CUDA_SUCCESS)                                              \
{                                                                   \
    printf ("%s:%d: error %d for CUDA Driver API function '%s'\n",    \
            __FILE__, __LINE__, err, cufunc);                         \
    exit(-1);                                                         \
}

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
    if (err != CUPTI_SUCCESS)                                             \
{                                                                   \
    const char *errstr;                                               \
    cuptiGetResultString(err, &errstr);                               \
    printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",          \
            __FILE__, __LINE__, errstr, cuptifunc);                   \
    exit(-1);                                                         \
}

void start_event_collection(){
    CUptiResult cuptiErr;
    CUresult err;
    CUcontext context = 0;
    CUdevice dev = 0;
    CUcontext context_g = 0;
    int maxDev;

    int deviceNum = 0;

    const char eventNameWarps[] = {'a','c','t','i','v','e','_','w','a','r','p','s','\0'};
    const char eventNameCycles[] = {'a','c','t','i','v','e','_','c','y','c','l','e','s','\0'};

    // err = cuDeviceGet(&dev, deviceNum);
    // CHECK_CU_ERROR(err, "cuDeviceGet");

    cudaGetDevice(&dev);

    if(!cuptiEventSet){
        cudaGetDeviceCount(&maxDev);
        cuptiEvent= (cupti_eventData*)malloc(maxDev*sizeof(cupti_eventData));
    }

    /* creating context */
    err = cuCtxCreate(&context, 0, dev);
    CHECK_CU_ERROR(err, "cuCtxCreate");
    context_g = context;

    /* Creating event group for profiling */
    cuptiErr = cuptiEventGroupCreate(context, &cuptiEvent[dev].eventGroup, 0);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

    cuptiErr = cuptiEventGetIdFromName(dev, eventNameWarps, &idWarps);
    if (cuptiErr != CUPTI_SUCCESS) 
    { 
        printf("Invalid eventName: %s\n", eventNameWarps);
        return; 
    }
    cuptiErr = cuptiEventGetIdFromName(dev, eventNameCycles, &idCycles);
    if (cuptiErr != CUPTI_SUCCESS) 
    { 
        printf("Invalid eventName: %s\n", eventNameCycles);
        return; 
    }

    /* adding events to the profiling group */
    cuptiErr = cuptiEventGroupAddEvent(cuptiEvent[dev].eventGroup, idWarps);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

    cuptiErr = cuptiEventGroupAddEvent(cuptiEvent[dev].eventGroup, idCycles);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

#ifdef SYNCHRONIZE_WITH_EVENT
    cudaEventRecord(start_event, 0);
    cudaDeviceSynchronize();
#endif
    cuptiErr = cuptiSetEventCollectionMode(context_g, 
            CUPTI_EVENT_COLLECTION_MODE_KERNEL);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");

    cuptiErr = cuptiEventGroupEnable(cuptiEvent[dev].eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");

    printf("Setted device %d\n",dev);
}

void stop_event_collection(){
    CUptiResult cuptiErr;
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceSynchronize();
    cuptiErr = cuptiEventGroupDisable(cuptiEvent[dev].eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");

}

void get_occupancy(){

    CUptiResult cuptiErr;
    size_t bytesRead;
    uint64_t activeWarps;
    uint64_t activeCycles;
    int maxWarps;

    CUdevice dev1;

    cudaGetDevice(&dev1);

#ifdef SYNCHRONIZE_WITH_EVENT/*uses events to synchronize with the kernel execution*/
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
#endif

    bytesRead = sizeof (uint64_t);
    //cudaDeviceSynchronize();
    cuptiErr = cuptiEventGroupReadEvent(cuptiEvent[dev1].eventGroup, 
            CUPTI_EVENT_READ_FLAG_NONE, 
            idWarps, 
            &bytesRead, &activeWarps);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");

    cuptiErr = cuptiEventGroupReadEvent(cuptiEvent[dev1].eventGroup, 
            CUPTI_EVENT_READ_FLAG_NONE, 
            idCycles, 
            &bytesRead, &activeCycles);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");

    cudaDeviceProp prop;
    CUdevice device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    if(!activeCycles){
        printf("Active cycles: %f\nActive warps: %f, on device %d\n",(double)activeCycles,(double)activeWarps,dev1);
    }else{
        printf("Achieved occupancy: %f%c, on device %d\n",((double)activeWarps / (double)activeCycles) / (double)maxWarps * 100,37,dev1 );
    }

}

void * sampling_func(void *arg){

    uint64_t valueBuffer[2];
    size_t eventValueBufferSize = 2*sizeof(uint64_t);
    size_t eventIdArraySize = 2*sizeof(uint32_t);
    size_t numEventIdsRead;
    int maxWarps;
    CUpti_EventID eventIds[2];
    CUptiResult cuptiErr;

    eventIds[0] = idWarps;
    eventIds[1] = idCycles;

    cudaDeviceProp prop;
    CUdevice device;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    while (!testComplete) {
        cuptiErr = cuptiEventGroupReadAllEvents(cuptiEvent[device].eventGroup,
                CUPTI_EVENT_READ_FLAG_NONE,
                &eventValueBufferSize,
                valueBuffer,
                &eventIdArraySize,
                eventIds,
                &numEventIdsRead);

        //cuptiErr = cuptiDeviceGetTimestamp(context, &timeStamp);

        CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadAllEvents");
        double occupancy = 0;
        if(valueBuffer[1]){
            occupancy = (double)valueBuffer[0] / (double)valueBuffer[1] / maxWarps*100;
        }
        printf("%f\n",occupancy);/*todo cant use printf*/

        //timeStampOld = timeStamp;
        usleep(1000);
    }
    return NULL;
}

void start_sampling(){
    int status = 0;
    pthread_t pThread;
    testComplete = 0;
    status = pthread_create(&pThread, NULL, sampling_func, NULL);
    if (status != 0) {
        perror("pthread_create");
        exit(-1);
    }
}

void stop_sampling(){
    //printf("Stop sampling thread\n");
    testComplete = 1;
}
