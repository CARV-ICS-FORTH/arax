#ifndef VINE_OBJECT_HEADER
#define VINE_OBJECT_HEADER
#include "utils/list.h"
#include "utils/spinlock.h"

#define VINE_OBJECT_NAME_SIZE 32

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
	VINE_DATA_COMPLETION,	/* Completions  */
	VINE_TYPE_COUNT			/* Number of types */
} vine_object_type_e;

/**
 * Vine object repository struct, contains references to all Vineyard objects.
 */
typedef struct {
	struct {
		utils_list_s   list;
		utils_spinlock lock;
	} repo[VINE_TYPE_COUNT];
} vine_object_repo_s;

/**
 * Vine Object super class, all Vine Objects have this as their first member.
 */
typedef struct {
	utils_list_node_s  list;
	vine_object_type_e type;
	char               name[VINE_OBJECT_NAME_SIZE];
} vine_object_s;

/**
 * Initialize an vine_object_repo_s instance on allocated pointer \c repo.
 *
 * \param repo An atleast sizeof(vine_object_repo_s) big buffer.
 */
void vine_object_repo_init(vine_object_repo_s *repo);

/**
 * Perform cleanup and exit time checks.
 *
 * Prints on stderr, the Objects still registered and thus considered as
 * leaks.
 *
 * \param repo A valid vine_object_repo_s instance.
 * \return Number of 'leaked' Vine Objects.
 */
int vine_object_repo_exit(vine_object_repo_s *repo);

/**
 * Vine Object 'Constructor'
 *
 * Register \c obj at \c repo.
 *
 * \param repo A valid vine_object_repo_s instance.
 * \param obj The object to be registered.
 * \param type Type of the new vine_object.
 * \param name The name on the new vine_object.
 */
void vine_object_register(vine_object_repo_s *repo, vine_object_s *obj,
                          vine_object_type_e type, const char *name);

/**
 * Remove \c obj from \c repo.
 *
 * \param repo A valid vine_object_repo_s instance.
 * \param obj The object to be removed from the repo.
 *
 */
void vine_object_remove(vine_object_repo_s *repo, vine_object_s *obj);

/**
 * Get a locked utils_list_s for traversing all objects of \c type.
 *
 * \param repo A valid vine_object_repo_s instance.
 * \param type Type of objects contained in list.
 */
utils_list_s* vine_object_list_lock(vine_object_repo_s *repo,
                                      vine_object_type_e type);

/**
 * Unlock previously locked list.
 * Parameters should be the same with the previous vine_object_list_locked()
 * invocation.
 *
 * \param repo A valid vine_object_repo_s instance.
 * \param type Type of objects contained in list.
 */
void vine_object_list_unlock(vine_object_repo_s *repo, vine_object_type_e type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef VINE_OBJECT_HEADER */
