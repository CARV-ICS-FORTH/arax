#include "async.h"
#include "utils/config.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

enum RegisterOffsets
{
	ISR_REG = 0,
	IMR_REG = 1,
	VM_ID_REG = 2,
	BELL_REG = 3
};

struct ivshmem
{
	volatile unsigned int regs[64];
};

void wakeupVm(async_meta_s * meta,unsigned int vm_id)
{
	meta->regs->regs[BELL_REG] = (vm_id<<16)|0; //Wakeup
}

unsigned int getVmID(async_meta_s * meta)
{
	return meta->regs->regs[VM_ID_REG];
}

void * async_thread(void * data)
{
	async_meta_s * meta = data;
	int buff;
	printf("async_thread started (VM:%d)!\n",meta->regs->regs[VM_ID_REG]);
	while(meta->fd)
	{
		printf("Waiting!\n");
		// Waiting for 'interrupt'
		read(meta->fd,&buff,sizeof(buff));
		printf("Wow something happened!\n");
	}
	return 0;
}

void async_meta_init(async_meta_s * meta)
{
	size_t shm_off     = 0;
	int    shm_ivshmem = 0;
	char   shm_file[1024];

	if ( !util_config_get_str("shm_file", shm_file, 1024) ) {
		fprintf(stderr,"No shm_file config line specified!\n");
		abort();
	}
	util_config_get_size("shm_off", &shm_off, 0);
	util_config_get_bool("shm_ivshmem", &shm_ivshmem, 0);

	if(!shm_ivshmem)
	{
		fprintf(stderr,"Attempted to use ivshmem synchronization on non-"
					   "ivshmem setup!\n");
		abort();
	}

	meta->fd = open(shm_file, O_CREAT|O_RDWR, 0644);

	if(meta->fd <= 0)
	{
		fprintf(stderr,"Failed to open shm_file: %s\n",shm_file);
		abort();
	}

	utils_spinlock_init(&(meta->lock));
	utils_list_init(&(meta->outstanding));

	meta->regs = mmap(0, 256, PROT_READ|PROT_WRITE,
						   MAP_SHARED, meta->fd, 0);

	if(pthread_create(&(meta->thread),0,async_thread,meta))
	{
		fprintf(stderr,"Failed to spawn async_thread\n");
		abort();
	}
}

void async_completion_init(async_meta_s * meta,async_completion_s * completion)
{
	completion->vm_id = getVmID(meta);
	utils_list_node_init(&(completion->outstanding),completion);
	pthread_mutexattr_init(&(completion->attr));
	pthread_mutexattr_setpshared(&(completion->attr), PTHREAD_PROCESS_SHARED);
	pthread_mutex_init(&(completion->mutex),&(completion->attr));
	pthread_mutex_lock(&(completion->mutex));
}

void async_completion_complete(async_meta_s * meta,async_completion_s * completion)
{
	completion->completed = 1; // Mark as completed
	meta->regs->regs[BELL_REG] = ((meta->regs->regs[VM_ID_REG])<<16)|0; //Wakeup
}

void async_completion_wait(async_meta_s * meta,async_completion_s * completion)
{
	utils_spinlock_lock(&(meta->lock));
	utils_list_add(&(meta->outstanding),&(completion->outstanding));
	utils_spinlock_unlock(&(meta->lock));
	pthread_mutex_lock(&(completion->mutex)); // Will sleep since already locked.
}

int async_completion_check(async_meta_s * meta,async_completion_s * completion)
{
	return completion->completed;
}

void async_meta_exit(async_meta_s * meta)
{
	int fd = meta->fd;

	meta->fd = 0;
	meta->regs->regs[BELL_REG] = ((meta->regs->regs[VM_ID_REG])<<16)|0; //Wakeup
	wakeupVm(meta,getVmID(meta));
	printf("Waiting for async_thread to exit!\n");
	pthread_join(meta->thread,0);
	munmap(meta->regs,256);
	close(fd);
}
