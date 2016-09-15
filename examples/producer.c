#include "vine_talk.h"
#include <stdlib.h>
#include <pthread.h>

/*Random define just to fill some blanks */
#define INPUT_SIZE  5
#define OUTPUT_SIZE 10
#define SIZEOF(ARG) 5

#define ADD_BYTE_CODE_SIZE 10

char *add_byte_code[ADD_BYTE_CODE_SIZE];
void* thread(void *thread_args)
{
	vine_proc *add_proc = vine_proc_get(CPU, "add"); /* Request function
	                                                  * from vineyard
	                                                  * process/function
	                                                  * repository. */

	if (!add_proc) /* Repository did not contain function */
		add_proc = vine_proc_register(CPU, "add", add_byte_code,
		                              ADD_BYTE_CODE_SIZE);  /* Register
	                                                             * function
	                                                             * to
	                                                             * vineyard
	                                                             * process/function
	                                                             * repository
	                                                             * and get
	                                                             * vine_proc
	                                                             * reference.
	                                                             * */


	vine_buffer_s inputs[2]=
	{
		VINE_BUFFER("Talk",INPUT_SIZE),
		VINE_BUFFER("Vine",INPUT_SIZE)
	};


	/* Input data initialized */
	char result[OUTPUT_SIZE] = {'F','A','I','L',0};
	vine_buffer_s  outputs[1] = {VINE_BUFFER(result,OUTPUT_SIZE)};
	vine_buffer_s *args = 0;
	vine_accel **accels;
	int        accels_count;

	accels_count = vine_accel_list(CPU, 1, &accels); /* Find all
	                                                 * usable/appropriate
	                                                 * physical accelerators. */

	if (!accels_count)
		return 0; /* No accelerators available! */

	printf("Found %d accelerators.\n", accels_count);

	vine_accel *accel; /* The accelerator to use */

	accel = accels[rand()%accels_count]; /* Choose accelerator randomly */

	vine_task *task = vine_task_issue(accel, add_proc, args, 2, inputs, 1,
	                                  outputs); /* Issue task to
	                                             * accelerator. */

	printf("Waiting for issued task %p.\n", task);

	if (vine_task_wait(task) == task_failed) /* Wait for task or exit if it
		                                  * fails */
		return 0;

	printf("Got: \'%s\'\n",result);

	vine_task_free(task);
	vine_proc_put(add_proc); /* Notify repository that add_proc is no longer
	                          * in use by us. */
	free(accels);

	return 0;
}                  /* thread */

void spawn_producers(pthread_t * threads,size_t number_of_threads)
{
	size_t i;
	for (i = 0; i < number_of_threads; i++) {
		pthread_create(threads, NULL, thread, NULL);
		threads++;
	}
}

void wait_producers(pthread_t * threads,size_t number_of_threads)
{
	size_t i;
	for (i = 0; i < number_of_threads; i++) {
		pthread_join(*threads, NULL);
		threads++;
	}
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		fprintf(stdout, "./ex2 <number_of_threads> \n");
		return 0;
	}

	int       number_of_threads = atoi(argv[1]);

	pthread_t tid[number_of_threads];

	spawn_producers(tid,number_of_threads);

	wait_producers(tid,number_of_threads);

	return 0;
}
