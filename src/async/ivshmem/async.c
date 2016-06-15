#include "async.h"
#include "utils/config.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

enum RegisterOffsets
{
	ISR_REG,
	IMR_REG,
	VM_ID_REG,
	BELL_REG
};

struct ivshmem
{
	unsigned int regs[256];
};

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
		fprintf(stderr,"Atempted to use ivshmem synchorniation on non-"
					   "ivshmem setup!\n");
		abort();
	}

	meta->fd = open(shm_file, O_CREAT|O_RDWR, 0644);

	if(meta->fd <= 0)
	{
		fprintf(stderr,"Failed to open shm_file: %s\n",shm_file);
		abort();
	}

	meta->regs = mmap(0, 4096, PROT_READ|PROT_WRITE|PROT_EXEC,
						   MAP_SHARED, meta->fd, 0);

}

void async_completion_init(async_completion_s * completion)
{
	completion->counter = 0;
}

void async_completion_complete(async_completion_s * completion)
{
	__sync_bool_compare_and_swap(&(completion->counter), 0, 1);
}

void async_completion_wait(async_completion_s * completion)
{
	while (!completion->counter);
}

void async_meta_exit(async_meta_s * meta)
{
	munmap(meta->regs,4096);
	close(meta->fd);
}
