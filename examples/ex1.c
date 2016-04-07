#include <vine_talk.h>
#include <stdlib.h>

/*Random define just to fill some blanks */
#define INPUT_SIZE  5
#define OUTPUT_SIZE 5
#define SIZEOF(ARG) 5

#define ADD_BYTE_CODE_SIZE 10

char *add_byte_code[ADD_BYTE_CODE_SIZE];

int main()
{
	int       i;
	vine_proc *add_proc = vine_proc_get(CPU, "add"); /* Request function
	                                                  * from vineyard
	                                                  * process/function
	                                                  * repository. */

	if (!add_proc) /* Repository did not contain function */
		add_proc = vine_proc_register(CPU, "add", add_byte_code,
		                              ADD_BYTE_CODE_SIZE); /* Register
		                                                    * function
		                                                    * to
		                                                    * vineyard
		                                                    * process/function
		                                                    * repository
		                                                    * and get
		                                                    * vine_proc
		                                                    * reference.
		                                                    * */


	vine_data *inputs[2];

	inputs[0] = vine_data_alloc(INPUT_SIZE, Both); /* Allocate space
	                                                * accessible from CPU
	                                                * and GPU for input */
	inputs[1] = vine_data_alloc(INPUT_SIZE, Both); /* Allocate space
	                                                * accessible from CPU
	                                                * and GPU for input */

	/* Initialize input data */
	for (i = 0; i < 2; i++) {
		void *data = vine_data_deref(inputs[i]); /* Get CPU usable
		                                          * pointer to data. */
		/* Fill data with user supplied input. */
	}

	/* Input data initialized */

	vine_data  *outputs[1] = {
		vine_data_alloc(OUTPUT_SIZE, Both)
	};         /* Allocate space accessible from CPU and GPU for input*/
	vine_data  *args       = 0;
	vine_accel **accels;

	int        accels_count;

	accels_count = vine_accel_list(CPU, &accels); /* Find all
	                                               * usable/appropriate
	                                               * accelerators. */

	if (!accels_count)
		return -1; /* No accelerators available! */

	printf("Found %d accelerators.\n", accels_count);

	vine_accel *accel; /* The accelerator to use */

	accel = accels[rand()%accels_count]; /* Choose accelerator randomly */

	vine_task *task = vine_task_issue(accel, add_proc, args, 2, inputs, 1,
	                                  outputs); /* Issue task to
	                                             * accelerator. */

	if (vine_task_wait(task) == task_failed) /* Wait for task or exit if it
		                                  * fails */
		return -1;

	void *result = vine_data_deref(outputs[0]); /* Get CPU usable pointer to
	                                             * result data. */

	/* Release data buffers */
	vine_data_free(inputs[0]);
	vine_data_free(inputs[1]);
	vine_data_free(outputs[0]);

	vine_proc_put(add_proc); /* Notify repository that add_proc is no longer
	                          * in use by us. */
	for(i = 0; i < accels_count; ++i){
		free(accels[i]);
	}
	free(accels);

	return 0;
} /* main */
