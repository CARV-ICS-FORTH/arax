#ifndef VINE_OBJECT_HEADER
#define VINE_OBJECT_HEADER
#include "utils/list.h"
#include "utils/spinlock.h"

#define VINE_OBJECT_NAME_SIZE 32

typedef enum vine_object_type{
	VINE_TYPE_PHYS_ACCEL,
	VINE_TYPE_VIRT_ACCEL,
	VINE_TYPE_PROC,
	VINE_TYPE_DATA,
	VINE_TYPE_COUNT
}vine_object_type_e;

typedef struct
{
	struct
	{
		utils_list_s list;
		utils_spinlock lock;
	}repo[VINE_TYPE_COUNT];
}vine_object_repo_s;

typedef struct {
	utils_list_node_s list;
	vine_object_type_e type;
	char name[VINE_OBJECT_NAME_SIZE];
}vine_object_s;

void vine_object_repo_init(vine_object_repo_s * repo);
void vine_object_repo_exit(vine_object_repo_s * repo);

void vine_object_register(vine_object_repo_s * repo,vine_object_s * obj,vine_object_type_e type,const char * name);
void vine_object_remove(vine_object_repo_s * repo,vine_object_s * obj);

utils_list_s * vine_object_list_locked(vine_object_repo_s * repo,vine_object_type_e type);
void vine_object_list_unlock(vine_object_repo_s * repo,vine_object_type_e type);
#endif
