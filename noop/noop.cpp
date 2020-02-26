#include "vine_talk.h"
#include "core/vine_data.h"
#include "VineLibUtilsCPU.h"
#include <cstring>

#ifdef BUILD_SO
vine_task_state_e noop(vine_task_msg_s * msg)
{
	int l = vine_data_size(msg->io[0]);
	char * in = (char*)vine_data_deref(msg->io[0]);
	char * out = (char*)vine_data_deref(msg->io[1]);
	printf("Nooping input:\"%s\"\n",in);
	char * data = (char*)vine_data_deref(msg->io[0]);
	int c;
	l-=2;
	for(c = 0 ; l >= 0 ; l--,c++)
		out[c] = in[l];
	out[c] = 0;
	printf("Nooping output:\"%s\"\n",out);
	vine_data_modified(msg->io[1],SHM_SYNC);
	vine_task_mark_done(msg,task_completed);
	return task_completed;
}

VINE_PROC_LIST_START()
VINE_PROCEDURE("noop", CPU, noop, 0)
VINE_PROCEDURE("noop", GPU, noop, 0)
VINE_PROC_LIST_END()
#endif

#ifdef BUILD_MAIN
int main(int argc,char * argv[])
{
	vine_talk_init();
	vine_accel * accel = vine_accel_acquire_type(CPU);
	vine_proc * proc = vine_proc_get(CPU,"noop");
	char noop[5] = "NOOP";
	vine_task * task;
	vine_buffer_s io[1] = {VINE_BUFFER(&noop, strlen(noop)+1)};

	vine_data_modified(io[0],USER_SYNC);

	vine_data_sync_to_remote(accel,io[0],0);

	task = vine_task_issue(accel, proc, 0, 0, 1, io, 1, io);

	vine_task_wait(task);

	vine_data_sync_from_remote(accel,io[0],1);
	
	fprintf(stderr,"Noop is \'%s\'\n",noop);
	vine_task_free(task);
	vine_proc_put(proc);
	vine_talk_exit();
	return 0;
}
#endif
