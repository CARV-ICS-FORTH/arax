#ifndef ASYNC_HEADER
#define ASYNC_HEADER
#include "core/arax_object.h"
#include "pthread.h"
#include <semaphore.h>

struct async_meta_s
{
    // This padp is necessary as empty struct have sizeof == 1 in C++, but 0 in C
    #ifndef __cplusplus
    char padd;
    #endif
};

typedef struct
{
    pthread_mutex_t     mutex;
    pthread_mutexattr_t attr;
}
async_completion_s;

typedef struct
{
    sem_t sem;
} async_semaphore_s;

typedef struct
{
    async_completion_s mutex;
    pthread_cond_t     condition;
    pthread_condattr_t c_attr;
} async_condition_s;

#include "async_api.h"
#endif // ifndef ASYNC_HEADER
