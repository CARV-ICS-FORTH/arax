#include "arax_object.h"
#include "arax_pipe.h"
#include "arax_accel.h"
#include "arax_proc.h"
#include "arax_task.h"
#include "arax_data.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static const char *type2str[ARAX_TYPE_COUNT] = {
    "Phys.Accel",// "Physical Accelerators",
    "Virt.Accel",// "Virtual Accelerators",
    "Procedures",
    "Arax--Data",
    "Arax-Tasks",
};

union arax_object_union {
    arax_accel_s    accel;
    arax_vaccel_s   vaccel;
    arax_proc_s     proc;
    arax_task_msg_s task;
    arax_data_s     data;
};

#ifdef ARAX_REF_DEBUG // if(OBJ->type==1)(specify which type of  object debug)
#define PRINT_REFS(OBJ, DELTA) \
    ({ \
        if ( (1 << (OBJ->type)) & ARAX_REF_DEBUG_MASK) \
        printf("%s:%s(%p,ABA:%d ,%d=>%d)\n", \
        __func__, type2str[OBJ->type], \
        OBJ, ((OBJ->ref_count & 0xffff0000) >> 16), \
        (OBJ->ref_count & 0xffff), \
        ((OBJ->ref_count & 0xffff) DELTA) & 0xffff); \
    })
#else
#define PRINT_REFS(OBJ, DELTA)
// without bitmask print
// #define PRINT_REFS(OBJ,DELTA)({ if(OBJ->type==3) printf("%s(%p(%s),%d=>%d)//\n",__func__,OBJ,type2str[OBJ->type],(OBJ->ref_count), (OBJ->ref_count DELTA)) ; } )
#endif /* ifdef ARAX_REF_DEBUG */

typedef void (*arax_object_dtor)(arax_pipe_s *pipe, arax_object_s *obj);

extern ARAX_OBJ_DTOR_DECL(arax_accel_s);
extern ARAX_OBJ_DTOR_DECL(arax_vaccel_s);
extern ARAX_OBJ_DTOR_DECL(arax_proc_s);
extern ARAX_OBJ_DTOR_DECL(arax_task_msg_s);
extern ARAX_OBJ_DTOR_DECL(arax_data_s);


static const arax_object_dtor dtor_table[ARAX_TYPE_COUNT] = {
    ARAX_OBJ_DTOR_USE(arax_accel_s),
    ARAX_OBJ_DTOR_USE(arax_vaccel_s),
    ARAX_OBJ_DTOR_USE(arax_proc_s),
    ARAX_OBJ_DTOR_USE(arax_data_s),
    ARAX_OBJ_DTOR_USE(arax_task_msg_s)
};

void arax_object_repo_init(arax_object_repo_s *repo, arax_pipe_s *pipe)
{
    int r;

    repo->pipe = pipe;
    for (r = 0; r < ARAX_TYPE_COUNT; r++) {
        utils_list_init(&repo->repo[r].list);
        utils_spinlock_init(&repo->repo[r].lock);
    }
}

int arax_object_repo_exit(arax_object_repo_s *repo)
{
    int r;
    int len;
    int failed = 0;

    for (r = 0; r < ARAX_TYPE_COUNT; r++) {
        len     = repo->repo[r].list.length;
        failed += len;
        if (len) {
            fprintf(stderr, "%lu %*s still registered!\n",
              repo->repo[r].list.length,
              (int) ( strlen(type2str[r]) - (len == 1) ),
              type2str[r]);
        }
    }
    return failed;
}

const char* arax_object_type_to_str(arax_object_type_e type)
{
    if (type < ARAX_TYPE_COUNT)
        return type2str[type];

    return 0;
}

arax_object_s* arax_object_register(arax_object_repo_s *repo,
  arax_object_type_e type, const char *name, size_t size, const int ref_count)
{
    arax_object_s *obj;

    arax_pipe_size_dec(repo->pipe, size);
    obj = arch_alloc_allocate(&(repo->pipe->allocator), size);

    if (!obj)      // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    obj->name = arch_alloc_allocate(&(repo->pipe->allocator), strlen(name) + 1);
    strcpy(obj->name, name);
    obj->repo       = repo;
    obj->alloc_size = size;
    obj->type       = type;
    obj->ref_count  = ref_count;
    utils_list_node_init(&(obj->list), obj);
    utils_spinlock_lock(&(repo->repo[type].lock) );
    utils_list_add(&(repo->repo[type].list), &(obj->list) );
    utils_spinlock_unlock(&(repo->repo[type].lock) );

    if (sizeof(union arax_object_union) >= size)
        memset(obj + 1, 0, size - sizeof(arax_object_s));
    else
        memset(obj + 1, 0, sizeof(union arax_object_union) - sizeof(arax_object_s));

    return obj;
}

void arax_object_rename(arax_object_s *obj, const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    char tmp;
    size_t new_size = vsnprintf(&tmp, 1, fmt, args);
    size_t old_size = strlen(obj->name);

    if (old_size < new_size) {
        arch_alloc_free(&(obj->repo->pipe->allocator), obj->name);
        obj->name = arch_alloc_allocate(&(obj->repo->pipe->allocator), new_size + 1);
    }
    vsprintf(obj->name, fmt, args);
    va_end(args);
}

void arax_object_ref_inc(arax_object_s *obj)
{
    arax_assert(obj);
    arax_assert(obj->type < ARAX_TYPE_COUNT);

    #ifdef ARAX_REF_DEBUG
    PRINT_REFS(obj, +0x10001);
    arax_assert( (obj->ref_count & 0xffff ) >= 0);
    __sync_add_and_fetch(&(obj->ref_count), 0x10001);
    #else
    PRINT_REFS(obj, +1);
    arax_assert(obj->ref_count >= 0);
    __sync_add_and_fetch(&(obj->ref_count), 1);
    #endif
}

int arax_object_ref_dec(arax_object_s *obj)
{
    arax_object_repo_s *repo;

    arax_assert(obj);
    arax_assert(obj->type < ARAX_TYPE_COUNT);

    repo = obj->repo;
    #ifdef ARAX_REF_DEBUG
    PRINT_REFS(obj, +0xffff);
    arax_assert( (obj->ref_count & 0xffff ) > 0);
    #else
    PRINT_REFS(obj, -1);
    arax_assert(obj->ref_count > 0);
    #endif

    utils_spinlock_lock(&(repo->repo[obj->type].lock) );

    #ifdef ARAX_REF_DEBUG
    int refs = __sync_add_and_fetch(&(obj->ref_count), 0xffff) & 0xffff;
    #else
    int refs = __sync_add_and_fetch(&(obj->ref_count), -1);
    #endif
    if (!refs) { // Seems to be no longer in use, must free it
        #ifdef ARAX_REF_DEBUG
        if (refs == (obj->ref_count & 0xffff ))
        #else
        if (refs == obj->ref_count)
        #endif
        {                                                                 // Ensure nobody changed the ref count
            utils_list_del(&(repo->repo[obj->type].list), &(obj->list) ); // remove it from repo
        }
        utils_spinlock_unlock(&(repo->repo[obj->type].lock) );

        dtor_table[obj->type](repo->pipe, obj);

        size_t size = obj->alloc_size;

        arch_alloc_free(&(repo->pipe->allocator), obj->name);

        arch_alloc_free(&(repo->pipe->allocator), obj);

        arax_pipe_size_inc(repo->pipe, size);
    } else {
        utils_spinlock_unlock(&(repo->repo[obj->type].lock) );
    }

    return refs;
} /* arax_object_ref_dec */

int arax_object_ref_dec_pre_locked(arax_object_s *obj)
{
    #ifdef ARAX_REF_DEBUG
    int refs = __sync_add_and_fetch(&(obj->ref_count), 0xffff) & 0xffff;
    #else
    int refs = __sync_add_and_fetch(&(obj->ref_count), -1);
    #endif
    if (!refs) { // Seems to be no longer in use, must free it
        arax_object_repo_s *repo = obj->repo;
        utils_list_del(&(repo->repo[obj->type].list), &(obj->list) ); // remove it from repo
        dtor_table[obj->type](repo->pipe, obj);

        size_t size = obj->alloc_size;

        arch_alloc_free(&(repo->pipe->allocator), obj->name);

        arch_alloc_free(&(repo->pipe->allocator), obj);

        arax_pipe_size_inc(repo->pipe, size);
    }

    arax_assert(refs >= 0);

    return refs;
}

int arax_object_refs(arax_object_s *obj)
{
    #ifdef ARAX_REF_DEBUG
    return (obj->ref_count & 0xffff);

    #else
    return obj->ref_count;

    #endif
}

utils_list_s* arax_object_list_lock(arax_object_repo_s *repo,
  arax_object_type_e                                    type)
{
    utils_spinlock_lock(&(repo->repo[type].lock) );
    return &(repo->repo[type].list);
}

void arax_object_list_unlock(arax_object_repo_s *repo, arax_object_type_e type)
{
    utils_spinlock_unlock(&(repo->repo[type].lock) );
}
