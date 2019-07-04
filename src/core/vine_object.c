#include "vine_object.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static const char *type2str[VINE_TYPE_COUNT] = {
	"Physical Accelerators",
	"Virtual Accelerators",
	"Procedures",
	"Tasks",
	"VineData "
};


typedef void (*vine_object_dtor)(vine_object_s *obj);

extern VINE_OBJ_DTOR_DECL(vine_accel_s);
extern VINE_OBJ_DTOR_DECL(vine_vaccel_s);
extern VINE_OBJ_DTOR_DECL(vine_proc_s);
extern VINE_OBJ_DTOR_DECL(vine_task_msg_s);
extern VINE_OBJ_DTOR_DECL(vine_data_s);


static const vine_object_dtor dtor_table[VINE_TYPE_COUNT] = {
	VINE_OBJ_DTOR_USE(vine_accel_s),
	VINE_OBJ_DTOR_USE(vine_vaccel_s),
	VINE_OBJ_DTOR_USE(vine_proc_s),
	VINE_OBJ_DTOR_USE(vine_data_s),
	VINE_OBJ_DTOR_USE(vine_task_msg_s)
};

void vine_object_repo_init(vine_object_repo_s *repo,arch_alloc_s *alloc)
{
	int r;

	repo->alloc = alloc;
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
					(int)( strlen(type2str[r])-(len == 1) ),
					type2str[r]);
	}
	return failed;
}

vine_object_s * vine_object_register(vine_object_repo_s *repo,
						  vine_object_type_e type, const char *name,size_t size)
{
	vine_object_s * obj;

	obj = arch_alloc_allocate(repo->alloc,size);

	if(!obj)
		return 0;

	memset(obj,0,size);

	snprintf(obj->name, VINE_OBJECT_NAME_SIZE, "%s", name);
	obj->repo = repo;
	obj->type = type;
	obj->ref_count = 1;
	utils_list_node_init(&(obj->list),obj);
	utils_spinlock_lock( &(repo->repo[type].lock) );
	utils_list_add( &(repo->repo[type].list), &(obj->list) );
	utils_spinlock_unlock( &(repo->repo[type].lock) );

	return obj;
}

void vine_object_rename(vine_object_s * obj,const char * fmt, ... )
{
	va_list args;
	va_start (args, fmt);
	vsnprintf (obj->name,VINE_OBJECT_NAME_SIZE,fmt, args);
}

void vine_object_ref_inc(vine_object_s * obj)
{
	__sync_add_and_fetch(&(obj->ref_count),1);
}

int vine_object_ref_dec(vine_object_s * obj)
{
	int refs = __sync_add_and_fetch(&(obj->ref_count),-1);

	if(!refs)
	{	// Seems to be no longer in use, must free it
		vine_object_repo_s * repo = obj->repo;
		utils_spinlock_lock( &(repo->repo[obj->type].lock) );
		if(refs == obj->ref_count)
		{	// Ensure nobody changed the ref count
			utils_list_del( &(repo->repo[obj->type].list), &(obj->list) );	//remove it from repo
		}
		utils_spinlock_unlock( &(repo->repo[obj->type].lock) );

		dtor_table[obj->type](obj);
	}

	return refs;
}

int vine_object_ref_dec_pre_locked(vine_object_s * obj)
{
	int refs = __sync_add_and_fetch(&(obj->ref_count),-1);

	if(!refs)
	{	// Seems to be no longer in use, must free it
		vine_object_repo_s * repo = obj->repo;
		utils_list_del( &(repo->repo[obj->type].list), &(obj->list) );	//remove it from repo
		dtor_table[obj->type](obj);
	}

	return refs;
}

int vine_object_refs(vine_object_s *obj)
{
	return obj->ref_count;
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
