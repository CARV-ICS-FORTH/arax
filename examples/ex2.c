#include "vine_pipe.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

void        *shm = 0;
vine_pipe_s *vpipe;

#define SHM_NAME "test"
/* 128 Mb Shared segment */
#define SHM_SIZE 128*1024*1024
/* 128 slost in ring */
#define RING_SIZE 128

void prepare_shm();

int main(int argc, char *argv[])
{
	prepare_shm();

	vine_task_msg_s *msg;

	while (1) {
		do {
			printf( "Used:%d\n", queue_used_slots(vpipe->queue) );
			msg = (vine_task_msg_s*)queue_pop(vpipe->queue);
			sleep(1);
		} while (!msg);
		printf("Got Message %p!", msg);

		int         start = msg->in_count;
		int         end   = start + msg->out_count;
		int         out;
		vine_data_s *vdata;

		for (out = start; out < end; out++) {
			vdata =
			        offset_to_pointer(vine_data_s*, vpipe,
			                          msg->io[out]);
			vine_data_mark_ready(vdata);
		}

	}
	return 0;
}

void prepare_shm()
{
	int err = 0;
	/* Once we figure configuration we will get the shm size,name
	 * dynamically */
	int fd = 0;

	if (vpipe) /* Already initialized */
		return;

	fd = shm_open(SHM_NAME, O_CREAT|O_RDWR, S_IRWXU);

	if (fd < 0) {
		err = __LINE__;
		goto FAIL;
	}

	if ( ftruncate(fd, SHM_SIZE) ) {
		err = __LINE__;
		goto FAIL;
	}

	do {
		shm = mmap(shm, SHM_SIZE, PROT_READ|PROT_WRITE|PROT_EXEC,
		           MAP_SHARED|(shm ? MAP_FIXED : 0), fd, 0);

		if (!shm || shm == MAP_FAILED) {
			err = __LINE__;
			goto FAIL;
		}

		vpipe = vine_pipe_init(shm, SHM_SIZE, RING_SIZE);
		shm   = vpipe->self; /* This is where i want to go */

		if (vpipe != vpipe->self) {
			printf("Remapping from %p to %p.\n", vpipe,
			       vpipe->self);
			munmap(vpipe, SHM_SIZE);
		}
	} while (shm != vpipe); /* Not where i want */
	printf("ShmLocation:%p\n", shm);
	printf("ShmSize:%d\n", SHM_SIZE);

	{
		/* Make a dummy accelerator */
		vine_accel_s *accel = (vine_accel_s*)vine_alloc_alloc(
		         vpipe->allocator, vine_accel_calc_size("FakeAccel1") );

		vine_accel_init(accel, "FakeAccel1", CPU);
		vine_pipe_register_accel(vpipe, accel);
		return;
	}

FAIL:   printf("prepare_shm Failed on line %d (shm:%p)\n", err, shm);
	exit(0);
} /* prepare_shm */
