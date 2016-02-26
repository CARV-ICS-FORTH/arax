/**
 *
 * @file
 *
 * Example use:
 *
 * \code{.c}
 *	int cpu_devs = vine_accel_count(GPU_SOFT);	// Get number of GPU_SOFT devices
 *
 *	if(!cpu_devs)	// No devices!
 *		return;
 *
 *	int dev = rand()%cpu_devs;	// Pick one randomly
 *	int dev = -1;	// Let scheduler decide
 *
 *	vine_func * hello_func = vine_func_register(GPU_SOFT,"hello_world",hello_cuda); // Register function, with binaries produced by CUDA compiler.
 *	vine_data * hello_in[] = {vine_data_alloc(strlen("Hello World")+1)};	// Allocate input data on ivshm(and also on GPU?)
 *	vine_data * hello_out[] = {vine_data_alloc(RESULT_SIZE)};	// Allocate result data on (locally and remotely)
 *	char * data = vine_data_get(hello_in[0]);	// Get usable memory reference.
 *	sprintf(data,"%s","Hello World");	// Produce input data
 *
 *	vine_call * call = vine_call(dev,hello_func,hello_in,hello_out);	// Issue function function to accelerator
 *	void * result_data;
 *
 *	while(!(result_data = vine_data_get(hello_out[0])));	// Spin untile we get result[0]
 *
 *	result_data = vine_data_wait(hello_out[0]);	// Wait for it
 *
 *
 *	printf("Result:%s\n",result_data);	// Process output data
 *
 *	// Here we can either refeed results again to a different function
 *
 *	// Or finish with data and exit
 *	vine_data_free(hello_in);
 *	vine_data_free(hello_out);
 *	vine_func_free(hello_func);
 * \endcode
 */

#ifndef VINE_TALK
	#define VINE_TALK
	/**
	 * Accelerator type enumeration.
	 */
	typedef enum vine_accel_e
	{
		ANY,		///< Let Scheduler Decide
		GPU,		///< Run on GPU with CUDA
		GPU_SOFT,	///< Run on CPU with software CUDA(Useful for debug?)
		CPU,		///< Run Native x86 code
		FPGA		///< Custom Fpga accelerator
	}vine_accel_e;
	/**
	 * vine_func: Opaque function pointer.
	 */
	typedef vine_func void;
	/**
	 * Return number of accelerators of provided type
	 * If zero is returned no matching devices were found.
	 * Non zero value(X) means that device ids from 0 through X can be used.
	 *
	 * @param type Count only accelerators of specified vine_accel_e
	 * @return Number of available accelerators of specified type.
	 */
	int vine_accel_count(vine_accel_e type);

	/**
	 * Return location object for accelerator specified by dev/type.
	 *
	 * @param dev Accelerator id.
	 * @param type Accelerator type.
	 * @return vine_loc object specifying the location of an accelerator.
	 */
	vine_loc vine_accel_location(int dev,vine_accel_e type);

	/**
	 * Acquire accelerator specified by dev/type for exclusive access.
	 *
	 * @param dev Accelerator id.
	 * @param type Accelerator type.
	 * @return Return 1 if successful, 0 on failure.
	 *
	 */
	int vine_accel_acquire(int dev,vine_accel_e type);

	/**
	 * Release previously acquired accelerator.
	 *
	 * @param dev Accelerator id.
	 * @param type Accelerator type.
	 *
	 */
	void vine_accel_release(int dev,vine_accel_e type);

	/**
	 * Register a new function 'func_name' for vine_accel_e type accelerators.
	 * Returned vine_func * identifies given function globally.
	 * func_bytes contains executable for the given vine_accel_e.
	 * (e.g. for Fpga bitstream, for GPU CUDA binaries or source(?), for CPU binary code)
	 *
	 * In case a function is already registered(same type and func_name), func_bytes will be compared.
	 * If func_bytes are equal the function will return the previously registered vine_func.
	 * If func_bytes don't match, the second function will return NULL denoting failure.
	 *
	 * \note For every vine_func_get()/vine_func_register() there should be a
	 * matching call to vine_func_free()
	 *
	 * @param type Provided binaries work for this type of accelerators.
	 * @param func_name Descriptive name of function, has to be unique for given type.
	 * @param func_bytes Binary containing executable of the appropriate format.
	 * @return vine_func * corresponding to the registered function, NULL on failure.
	 */
	vine_func * vine_func_register(vine_accel_e type,const char * func_name,const char * func_bytes);
	/**
	 * Retrieve a previously registered vine_func pointer.
	 *
	 * \note For every vine_func_get()/vine_func_register() there should be a
	 * matching call to vine_func_free()
	 *
	 * @param type Provided binaries work for this type of accelerators.
	 * @param func_name Descriptive name of function, as provided to vine_func_register.
	 * @return vine_func * corresponding to the requested function, NULL on failure.
	 */
	vine_func * vine_func_get(vine_accel_e type,const char * func_name);
	/**
	 * Delete registered vine_func pointer.
	 *
	 * \note For every vine_func_get()/vine_func_register() there should be a
	 * matching call to vine_func_free()
	 *
	 * @param func vine_func to be deleted.
	 */
	int vine_func_free(vine_func * func);
	/**
	 * vine_data: Opaque data pointer.
	 */
	typedef vine_data void;

	/**
	 * Allocation strategy enumeration.
	 */
	typedef enum vine_data_alloc_place_e
	{
		HostOnly = 1,	///< Allocate space only on host memory(RAM)
		AccelOnly = 2,	///< Allocate space only on accelerator memory (e.g. GPU VRAM)
		Both = 3		///< Allocate space on both host memory and accelerator memory.
	}vine_data_alloc_place_e;

	/**
	 * Allocate data buffers necessary for a vine_call.
	 *
	 * @param size Size in bytes of the data buffer to be allocated.
	 * @param place Choose where data allocation occurs.
	 * @return Allocated vine_data pointer.NULL on failure.
	 */
	vine_data * vine_data_alloc(size_t size,vine_data_alloc_place_e place);

	/**
	 * Return size of provided vine_data object.
	 * @param data Valid vine_data pointer.
	 * @return Return size of data of provided vine_data object.
	 */
	size_t vine_data_size(vine_data * data);
	/**
	 * Get pointer to buffer for use from CPU.
	 *
	 * @param data Valid vine_data pointer.
	 * @return Ram point to vine_data buffer.NULL on failure.
	 */
	void * vine_data_get(vine_data * data);
	/**
	 * Release resources of given vine_data.
	 *
	 * @param data Allocated vine_data pointer to be deleted.
	 */
	void vine_data_free(vine_data * data);
	typedef vine_call void;
	/**
	 * Execute vine_func with given vine_data on specified dev.
	 *
	 * @param dev Device id.
	 * @param func vine_func to be dispatched on accelerator.
	 * @param input array of vine_data pointers with input data
	 * @param output array of vine_data pointers with output data
	 * @return vine_call * corresponding to the issued function invocation.
	 */
	vine_call * vine_call(int dev,vine_func * func,vine_data ** input,vine_data ** output);
	/**
	 * Wait for completion of vine_call and get usable pointer.
	 *
	 * @param data vine_data to wait until received.
	 * @return CPU usable pointer to the data.
	 */
	void * vine_data_wait(vine_data * data);
	/**
	 * Get usable pointer to data.
	 *
	 * @param data vine_data to receive.
	 * @return CPU usable pointer to the data.NULL if called before producer vine_call not completed.
	 */
	void * vine_data_get(vine_data * data);
#endif
