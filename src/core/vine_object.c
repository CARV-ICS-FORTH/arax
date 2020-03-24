#include "vine_object.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static const char *type2str[VINE_TYPE_COUNT] = {
	"Phys.Accel",//"Physical Accelerators",
	"Virt.Accel",//"Virtual Accelerators",
	"Procedures",
	"Vine-Tasks",
	"Vine--Data"
};

#ifdef VINE_REF_DEBUG //if(OBJ->type==1)(specify which type of vine object debug)
	#define PRINT_REFS(OBJ,DELTA)({ \
		printf("%s:%s(%p,ABA:%d ,%d=>%d)\n",__func__,type2str[OBJ->type],OBJ,(((OBJ->ref_count&0xffff0000)>>16)&0xffff) ,(OBJ->ref_count&0xffff), ((OBJ->ref_count&0xffff) DELTA)&0xffff);})
#else
	#define PRINT_REFS(OBJ,DELTA)
	//without bitmask print
	//#define PRINT_REFS(OBJ,DELTA)({ if(OBJ->type==3) printf("%s(%p(%s),%d=>%d)//\n",__func__,OBJ,type2str[OBJ->type],(OBJ->ref_count), (OBJ->ref_count DELTA)) ; } )
#endif

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
						  vine_object_type_e type, const char *name,size_t size,const int ref_count)
{
	vine_object_s * obj;

	obj = arch_alloc_allocate(repo->alloc,size);

	if(!obj)		// GCOV_EXCL_LINE
		return 0;	// GCOV_EXCL_LINE

	memset(obj,0,size);

	snprintf(obj->name, VINE_OBJECT_NAME_SIZE, "%s", name);
	obj->repo = repo;
	obj->type = type;
	obj->ref_count = ref_count;
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
	vine_assert(obj);

#ifdef VINE_REF_DEBUG
	PRINT_REFS(obj,+0x10001);
	vine_assert( (obj->ref_count & 0xffff )>= 0);
	__sync_add_and_fetch(&(obj->ref_count),0x10001);
#else
	PRINT_REFS(obj,+1);
	vine_assert(obj->ref_count >= 0);
	__sync_add_and_fetch(&(obj->ref_count),1);
#endif
}

int vine_object_ref_dec(vine_object_s * obj)
{
	vine_object_repo_s * repo;

	vine_assert(obj);

	repo = obj->repo;    
#ifdef VINE_REF_DEBUG
	PRINT_REFS(obj,+0xffff);
	vine_assert( (obj->ref_count& 0xffff ) >= 0);
#else
	PRINT_REFS(obj,-1);
	vine_assert(obj->ref_count >= 0);
#endif

	utils_spinlock_lock( &(repo->repo[obj->type].lock) );

#ifdef VINE_REF_DEBUG
	int refs = __sync_add_and_fetch(&(obj->ref_count),0xffff ) & 0xffff;
#else
    int refs = __sync_add_and_fetch(&(obj->ref_count),-1);
#endif
	if(!refs)
	{	// Seems to be no longer in use, must free it
#ifdef VINE_REF_DEBUG
		if(refs == (obj->ref_count &  0xffff ))
#else
		if(refs == obj->ref_count)
#endif
		{	// Ensure nobody changed the ref count
            utils_list_del( &(repo->repo[obj->type].list), &(obj->list) );	//remove it from repo
        }
		utils_spinlock_unlock( &(repo->repo[obj->type].lock) );

        dtor_table[obj->type](obj);
	}
	else
		utils_spinlock_unlock( &(repo->repo[obj->type].lock) );

	return refs;
}

int vine_object_ref_dec_pre_locked(vine_object_s * obj)
{
#ifdef VINE_REF_DEBUG
	int refs = __sync_add_and_fetch(&(obj->ref_count),0xffff) & 0xffff ;
#else
    int refs = __sync_add_and_fetch(&(obj->ref_count),-1);
#endif
	if(!refs)
	{	// Seems to be no longer in use, must free it
		vine_object_repo_s * repo = obj->repo;
        utils_list_del( &(repo->repo[obj->type].list), &(obj->list) );	//remove it from repo
        dtor_table[obj->type](obj);
    }

	vine_assert(refs >= 0);

	return refs;
}

int vine_object_refs(vine_object_s *obj)
{
#ifdef VINE_REF_DEBUG
	return (obj->ref_count & 0xffff) ;
#else
	return obj->ref_count;
#endif
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
