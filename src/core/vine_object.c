#include "vine_object.h"
#include "utils/spinlock.h"
#include <stdio.h>
#include <string.h>

static const char *type2str[VINE_TYPE_COUNT] = {
	"Physical Accelerators", "Virtual Accelerators", "Procedures",
	"VineData "
};

void vine_object_repo_init(vine_object_repo_s *repo)
{
	int r;

	for (r = 0; r < VINE_TYPE_COUNT; r++) {
		utils_list_init(&repo->repo[r].list);
		utils_spinlock_init(&repo->repo[r].lock);
	}
}

int vine_object_repo_exit(vine_object_repo_s *repo)
{
	int r;
	int len;
	int failed = 0;

	for (r = 0; r < VINE_TYPE_COUNT; r++) {
		len     = repo->repo[r].list.length;
		failed += len;
		if (len)
			fprintf(stderr, "%lu %*s still registered!\n",
			        repo->repo[r].list.length,
			        (int)( strlen(
			                       type2str[r])-(len == 1) ),
			        type2str[r]);
	}
	return failed;
}

void vine_object_register(vine_object_repo_s *repo, vine_object_s *obj,
                          vine_object_type_e type, const char *name)
{
	snprintf(obj->name, VINE_OBJECT_NAME_SIZE, "%s", name);
	obj->type = type;
	utils_spinlock_lock( &(repo->repo[type].lock) );
	utils_list_add( &(repo->repo[type].list), &(obj->list) );
	utils_spinlock_unlock( &(repo->repo[type].lock) );
}

void vine_object_remove(vine_object_repo_s *repo, vine_object_s *obj)
{
	utils_spinlock_lock( &(repo->repo[obj->type].lock) );
	utils_list_del( &(repo->repo[obj->type].list), &(obj->list) );
	utils_spinlock_unlock( &(repo->repo[obj->type].lock) );
}

utils_list_s* vine_object_list_lock(vine_object_repo_s *repo,
                                      vine_object_type_e type)
{
	utils_spinlock_lock( &(repo->repo[type].lock) );
	return &(repo->repo[type].list);
}

void vine_object_list_unlock(vine_object_repo_s *repo, vine_object_type_e type)
{
	utils_spinlock_unlock( &(repo->repo[type].lock) );
}
