#include <vine_pipe.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

void        *shm = 0;
vine_pipe_s *vpipe;

int main(int argc, char *argv[])
{
	vine_task_msg_s *msg;
	vpipe = vine_pipe_get();

	vine_accel_s * acc;
	acc = arch_alloc_allocate(
		vpipe->allocator,
		vine_accel_calc_size("FakeAccel1"));
	acc = vine_accel_init(acc,"FakeAccel1",CPU);
	vine_pipe_register_accel(vpipe,acc);

	while (1) {
		do {
			printf( "Used:%d\n", utils_queue_used_slots(vpipe->queue) );
			msg = (vine_task_msg_s*)utils_queue_pop(vpipe->queue);
			sleep(1);
		} while (!msg);
		printf("Got task (%p) %s(%s)!", msg, vine_accel_get_name(msg->accel),
			   ((vine_proc_s*)msg->proc)->name);

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
