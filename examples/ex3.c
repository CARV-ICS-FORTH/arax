#include "vine_talk.h"
#include <stdlib.h>
#include <pthread.h>

/*Random define just to fill some blanks */
#define INPUT_SIZE  5
#define OUTPUT_SIZE 5

int add_x86(int a, int b);


void* test(void *args)
{
	int i;
	/* Request function from vineyard process/function repository. */
	vine_proc *add_proc = vine_proc_get(CPU, "add");

	if (!add_proc) /* Repository did not contain function */
		/* Register function to vineyard process/function repository
		 * and get vine_proc reference. */
		add_proc =
		        vine_proc_register( CPU, "add", add_x86,
		                            sizeof(add_x86) );

	vine_data *inputs[2];

	/* Allocate space accessible from CPU and GPU for input */
	inputs[0] = vine_data_alloc(INPUT_SIZE, Both);
	/* Allocate space accessible from CPU and GPU for input */
	inputs[1] = vine_data_alloc(INPUT_SIZE, Both);

	/* Initialize input data */
	for (i = 0; i < 2; i++) {
		/* Get CPU usable pointer to data. */
		void *data = vine_data_deref(inputs[i]);
		/* Fill data with user supplied input. */
	}

	/* Input data initialized */

	/* Allocate space accessible from CPU and GPU for input */
	vine_data  *outputs[1] = {
		vine_data_alloc(OUTPUT_SIZE, Both)
	};
	vine_accel **accels;
	int        accels_count;

	/* Find all usable/appropriate accelerators. */
	accels_count = vine_accel_list(CPU, &accels);

	if (!accels_count) {
		printf("Error \n");
		exit(-1); /* kill */
	}

	vine_accel *accel; /* The accelerator to use */

	accel = accels[rand()%accels_count]; /* Choose accelerator randomly */

	/* Issue task to accelerator. */
	vine_task *task = vine_task_issue(accel, add_proc, NULL, 2*INPUT_SIZE,
	                                  inputs, OUTPUT_SIZE, outputs);

	if (vine_task_wait(task) == task_failed) {
		/* Wait for task or exit if
		 * it fails */
		printf("Error \n");
		exit(-1); /* kill */
	}

	/* Get CPU usable pointer to result data. */
	void *result = vine_data_deref(outputs[0]);

	/* Release data buffers */
	vine_data_free(inputs[0]);
	vine_data_free(inputs[1]);
	vine_data_free(outputs[0]);

	/* Notify repository that add_proc is no longer in use by us. */
	vine_proc_put(add_proc);
	return 0;
}                  /* test */

int main(int argc, char *argv[])
{
	if (argc != 2) {
		fprintf(stdout, "./ex2 <number_of_thread> \n");
		return 0;
	}

	int       number_of_threads = atoi(argv[1]);
	int       i;
	pthread_t tid[number_of_threads];

	for (i = 0; i < number_of_threads; i++) {
		pthread_create(&tid[i], NULL, test, NULL);
	}
	for (i = 0; i < number_of_threads; i++)
		pthread_join(tid[i], NULL);

	return 0;
}

int add_x86(int a, int b)
{
	return a+b;
}
