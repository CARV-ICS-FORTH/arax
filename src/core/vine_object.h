#ifndef VINE_OBJECT_HEADER
#define VINE_OBJECT_HEADER
#include "utils/list.h"
#include "utils/spinlock.h"
#include "arch/alloc.h"

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Enumeration with available Vine Object Types.
 */
typedef enum vine_object_type {
	VINE_TYPE_PHYS_ACCEL,	/* Physical Accelerator */
	VINE_TYPE_VIRT_ACCEL,	/* Virtual Accelerator */
	VINE_TYPE_PROC,			/* Procedure */
	VINE_TYPE_DATA,			/* Data Allocation */
	VINE_TYPE_TASK,			/* Task */
	VINE_TYPE_COUNT			/* Number of types */
} vine_object_type_e;

/**
 * Vine object repository struct, contains references to all Vineyard objects.
 */
typedef struct {
	arch_alloc_s *alloc;
	struct {
		utils_list_s   list;
		utils_spinlock lock;
	} repo[VINE_TYPE_COUNT];
} vine_object_repo_s;

/**
 * Vine Object super class, all Vine Objects have this as their first member.
 */
typedef struct {
	vine_object_repo_s * repo;
	utils_list_node_s  list;
	vine_object_type_e type;
	volatile int ref_count;
	char               name[VINE_OBJECT_NAME_SIZE];
} vine_object_s;

/**
 * Initialize an vine_object_repo_s instance on allocated pointer \c repo.
 *
 * @param repo An atleast sizeof(vine_object_repo_s) big buffer.
 * @param alloc Allocator instance to be used for any object allocation.
 */
void vine_object_repo_init(vine_object_repo_s *repo,arch_alloc_s *alloc);

/**
 * Perform cleanup and exit time checks.
 *
 * Prints on stderr, the Objects still registered and thus considered as
 * leaks.
 *
 * @param repo A valid vine_object_repo_s instance.
 * @return Number of 'leaked' Vine Objects.
 */
int vine_object_repo_exit(vine_object_repo_s *repo);

/**
 * Vine Object 'Constructor'
 *
 * Register \c obj at \c repo.
 *
 * Note: Sets reference count to 1
 *
 * @param repo A valid vine_object_repo_s instance.
 * @param type Type of the new vine_object.
 * @param name The name on the new vine_object.
 * @param size The size of the new object (sizeof(struct)).
 * @param ref_count Initialize ref count 
 */
vine_object_s * vine_object_register(vine_object_repo_s *repo,
									 vine_object_type_e type, const char *name,size_t size,const int ref_count);

/**
 * Change name of \c obj to printf like format \c fmt
 *
 * @param obj A valid vine_object_s instance.
 * @param fmt printf style format string
 * @param ... Args matching \c fmt
 */
void vine_object_rename(vine_object_s * obj,const char * fmt, ... );

/**
 * Increase reference count of \c obj.
 *
 * @param obj A valid vine_object_s instance.
 */
void vine_object_ref_inc(vine_object_s * obj);

/**
 * Decrease reference count of \c obj.
 *
 * @param obj A valid vine_object_s instance.
 * @return Reference count after decreasing, 0 means object was reclaimed
 */
int vine_object_ref_dec(vine_object_s * obj);

/**                                                                                                                                                           * Decrease reference count of \c obj dec_count times.                                                                                                        *                                                                                                                                                            * @param obj A valid vine_object_s instance.                                                                                                               
 * @param dec_count Decreace number for ref_counter   
 * @return Reference count after decreasing, 0 means object was reclaimed                                                                                     */  
int vine_object_ref_mul_dec(vine_object_s * obj,const int dec_count);

/**
 * Decrease reference count of \c obj.
 * \note Assumes object repo lock is held.
 *
 * @param obj A valid vine_object_s instance.
 * @return Reference count after decreasing, 0 means object was reclaimed
 */
int vine_object_ref_dec_pre_locked(vine_object_s * obj);

/**
 * Returns \c obj current reference count.
 *
 * @param obj A valid vine_object_s instance.
 */
int vine_object_refs(vine_object_s *obj);

/**
 * Get a locked utils_list_s for traversing all objects of \c type.
 *
 * @param repo A valid vine_object_repo_s instance.
 * @param type Type of objects contained in list.
 */
utils_list_s* vine_object_list_lock(vine_object_repo_s *repo,
                                      vine_object_type_e type);

/**
 * Unlock previously locked list.
 * Parameters should be the same with the previous vine_object_list_locked()
 * invocation.
 *
 * @param repo A valid vine_object_repo_s instance.
 * @param type Type of objects contained in list.
 */
void vine_object_list_unlock(vine_object_repo_s *repo, vine_object_type_e type);

#define VINE_OBJ_DTOR_DECL(TYPE) void __dtor_##TYPE(vine_object_s *obj)
#define VINE_OBJ_DTOR_USE(TYPE) __dtor_##TYPE

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_OBJECT_HEADER */
