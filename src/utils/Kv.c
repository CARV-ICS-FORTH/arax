#include "Kv.h"
#include <stdio.h>
#include <string.h>
#include "utils/vine_assert.h"

void utils_kv_init(utils_kv_s * kv)
{
	memset(kv,0,sizeof(*kv));
	utils_spinlock_init(&(kv->lock));
}

void utils_kv_set(utils_kv_s * kv,void * key,void * value)
{
	// TODO De-tarzanize implementation (ask christos for explanation)
	// use bsearch and qsort
	size_t itr = 0;
	utils_spinlock_lock(&(kv->lock));
	for(itr = 0 ; itr < kv->pairs ; itr++)
	{
		if(kv->kv[itr].key == key)
		{
			kv->kv[itr].value = value;
			utils_spinlock_unlock(&(kv->lock));
			return;
		}
	}
	if(kv->pairs < VINE_KV_CAP)
	{
		kv->kv[kv->pairs].key = key;
		kv->kv[kv->pairs++].value = value;
	}
	else
	{
		utils_spinlock_unlock(&(kv->lock));
		vine_assert(!"Exceeded VINE_KV_CAP");
	}
	utils_spinlock_unlock(&(kv->lock));
}

void ** utils_kv_get(utils_kv_s * kv,void * key)
{
	size_t itr;
	utils_spinlock_lock(&(kv->lock));
	for(itr = 0 ; itr < kv->pairs ; itr++)
	{
		if(kv->kv[itr].key == key)
		{
			utils_spinlock_unlock(&(kv->lock));
			return &(kv->kv[itr].value);
		}
	}
	utils_spinlock_unlock(&(kv->lock));
	return 0;
}
