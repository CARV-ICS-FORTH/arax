#include "conf.h"
#include "async.h"
#include "utils/config.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

// Taken from ivshmem manual/thesis:
// See Shared-Memory Optimizations for Virtual Machines by A. Cameron Macdonell
// Sections 4.6 (4.6.5 and 4.6.6 in particular)
enum RegisterOffsets
{
	ISR_REG = 0,	//< Not used(yet)
	IMR_REG = 1,	//< Not used(yet)
	VM_ID_REG = 2,	//< My VMs ID
	BELL_REG = 3	//< [DEST_VM_ID,FD_NUMBER]
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
static int reg_fd = 0;
void * async_thread(void * data)
{
	async_meta_s * meta = data;
	utils_list_node_s * itr;
	utils_list_node_s * tmp;
	async_completion_s * compl;
	int buff;
	printf("async_thread started (VM:%d)!\n",getVmID(meta));
	while(reg_fd)
	{
		// Waiting for 'interrupt'
		read(reg_fd,&buff,sizeof(buff));
		// something happened
		utils_list_for_each_safe(meta->outstanding,itr,tmp)
		{
			compl = itr->owner;
			// Should check completion is mine?
			if( async_completion_check(compl) )
			{
				utils_list_del(&(meta->outstanding),itr);
				pthread_mutex_unlock(&(compl->mutex));
			}
		}
	}
	printf("async_thread stoped (VM:%d)!\n",getVmID(meta));
	return 0;
}

void _add_completion(async_meta_s * meta,async_completion_s * completion)
{
	utils_spinlock_lock(&(meta->lock));
	utils_list_add(&(meta->outstanding),&(completion->outstanding));
	utils_spinlock_unlock(&(meta->lock));
}

void async_meta_init_once(async_meta_s * meta,arch_alloc_s * alloc)
{
	int    shm_ivshmem = 0;
	char   * config = utils_config_alloc_path(VINE_CONFIG_FILE);
	utils_config_get_bool(config,"shm_ivshmem", &shm_ivshmem, 0);
	utils_config_free_path(config);
	if(!shm_ivshmem)
	{
		fprintf(stderr,"Attempted to use ivshmem synchronization on non-"
		"ivshmem setup!\n");
		abort();
	}

	utils_spinlock_init(&(meta->lock));
	utils_list_init(&(meta->outstanding));
	meta->alloc = alloc;
}

void async_meta_init_always(async_meta_s * meta)
{
	char   shm_file[1024];
	char   * config = utils_config_alloc_path(VINE_CONFIG_FILE);

	if ( !utils_config_get_str(config,"shm_file", shm_file, 1024,0) ) {
		utils_config_free_path(config);
		abort();
	}
	utils_config_free_path(config);

	reg_fd = open(shm_file, O_CREAT|O_RDWR, 0644);

	if(reg_fd <= 0)
	{
		fprintf(stderr,"Failed to open shm_file: %s\n",shm_file);
		abort();
	}

	meta->regs = mmap(meta->regs, 256, PROT_READ|PROT_WRITE,
					  MAP_SHARED, reg_fd, 0);

	fprintf(stderr,"Ivshmem regs mapped at %p\n",meta->regs);

	if(pthread_create(&(meta->thread),0,async_thread,meta))
	{
		fprintf(stderr,"Failed to spawn async_thread\n");
		abort();
	}
}

void async_completion_init(async_meta_s * meta,async_completion_s * completion)
{
	completion->meta = meta;
	utils_list_node_init(&(completion->outstanding),completion);
	pthread_mutexattr_init(&(completion->attr));
	pthread_mutexattr_setpshared(&(completion->attr), PTHREAD_PROCESS_SHARED);
	pthread_mutex_init(&(completion->mutex),&(completion->attr));
	completion->completed = 0; 					// Completion not completed
	pthread_mutex_lock(&(completion->mutex));	// wait should block
}

void async_completion_complete(async_completion_s * completion)
{
	completion->completed = 1; // Mark as completed
	wakeupVm(completion->meta,completion->vm_id);
}

void async_completion_wait(async_completion_s * completion)
{
	if(pthread_mutex_trylock(&(completion->mutex)))
	{	// Failed, so add me to the outstanding list
		_add_completion(completion->meta,completion);
	}
	completion->vm_id = getVmID(completion->meta);
	pthread_mutex_lock(&(completion->mutex)); // Will sleep until mutex unlocked.
}

int async_completion_check(async_completion_s * completion)
{
	return completion->completed;
}

void async_semaphore_init(async_meta_s * meta,async_semaphore_s * sem)
{
	utils_list_init(&(sem->pending_list));
	utils_spinlock_init(&(sem->pending_lock));
	sem->value = 0;
}

int async_semaphore_value(async_semaphore_s * sem)
{
	return sem->value;
}

void async_semaphore_inc(async_semaphore_s * sem)
{
	int val = __sync_fetch_and_add(&(sem->value),1);
	utils_list_node_s * node,*next;
	// val contains the min number of penders that must wake up.
	if(sem->pending_list.length)
	{	// Seem to have penders
		utils_spinlock_lock(&(sem->pending_lock));
		do
		{	// Lets wake em
			node = sem->pending_list.head.next;
			next = node->next;
			utils_list_del(&(sem->pending_list),node);
			_add_completion(sem->meta,node->owner);
			async_completion_complete(node->owner); // Notify vm
			node = next;
		}while( (val = __sync_fetch_and_add(&(sem->value),-1)) );
		utils_spinlock_unlock(&(sem->pending_lock));
	}
}

void async_semaphore_dec(async_semaphore_s * sem)
{
	int val = sem->value;

	while(val) // Seems Positive
	{
		if(__sync_bool_compare_and_swap(&(sem->value),val-1,val))
		{	// Was positive and i got it
			return;
		}
		else
			val = sem->value;
	}
	// Have to wait
	async_completion_s * compl;
	compl = arch_alloc_allocate(sem->meta->alloc,sizeof(*compl));
	async_completion_init(sem->meta,compl);
	utils_spinlock_lock(&(sem->pending_lock));
	// Add is LIFO, this might be bad for tail latency(starvation).
	utils_list_add(&(sem->pending_list),&(compl->outstanding));
	utils_spinlock_unlock(&(sem->pending_lock));
	wakeupVm(sem->meta,getVmID(sem->meta));	// Might missed an inc
	async_completion_wait(compl);
	arch_alloc_free(sem->meta->alloc,compl);
}

void async_condition_init(async_meta_s * meta,async_condition_s * cond)
{
	async_completion_init(meta,&(cond->mutex));
	async_completion_complete(&(cond->mutex));
	async_semaphore_init(meta,&(cond->semaphore));
}

void async_condition_lock(async_condition_s * cond)
{
	async_completion_wait(&(cond->mutex));
}

void async_condition_wait(async_condition_s * cond)
{
	async_completion_complete(&(cond->mutex));
	async_semaphore_dec(&(cond->semaphore));
	async_completion_wait(&(cond->mutex));
}

void async_condition_notify(async_condition_s * cond)
{
	async_semaphore_inc(&(cond->semaphore));
}

void async_condition_unlock(async_condition_s * cond)
{
	async_completion_complete(&(cond->mutex));
}

void async_meta_exit(async_meta_s * meta)
{
	int fd = meta->fd;

	meta->fd = 0;
	wakeupVm(meta,getVmID(meta)); // Wakeup Myself to terminate async_thread
	printf("Waiting for async_thread to exit!\n");
	pthread_join(meta->thread,0);
	munmap(meta->regs,256);
	close(fd);
}
