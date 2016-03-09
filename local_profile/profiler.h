#ifndef PROFILER_H
	#define PROFILER_H

	#define		TRUE		1
	#define		FALSE		0
	#define		CONFIG_NAME	"profiler.conf"
	//#define DEACTIVATE_PROFILER 
 

	/**
	* useful for return value. 
	*/
	typedef unsigned bool;
	char*	curr_entry;
	char*	end_of_buffer_ptr;
	int     log_buffer_size;
	char* 	log_buffer_start_ptr ;   /*fixed size array of entries.*/ 
	int log_file ;
	bool	is_initialized = FALSE; 
	

	int create_log_entry(char* func_id,char* ret_status,int task_duration,
				vine_accel_type_e accel_type,unsigned incnt,unsigned outcnt,
				char** indata,char** outdata,char** new_entry_buffer);

	/**
	* Return true if pointer at the end of log_buffer 
	* is smaller than next entry end position pointer.
	* Otherwise returns false.
	* @param new_entry_bytes
	*
	* @return 0  == false,1 == true
	*/
	bool has_buffer_enough_space(int new_entry_bytes);

	void update_curr_entry(int size_of_new_entry);

	//char* get_next_entry_ptr(int size_of_new_entry);
	//void create_new_entry(char* func_id
	

	/**
	* Returns accel type as a string 
	*/
	char* enum_vine_accel_type_to_str(vine_accel_type_e type);

	/**
	* Returns the name of log buffer
	* which is in the form of this
	*  < trace_hostname_pid_date.csv >
	* @return  log file name
	*/
	char* get_log_file_name();

	/**
	* Allocate log_buffer with size bytes 
	* @param bytes
	*/
	void init_log_buffer(unsigned  bytes);


	/**
	* 1) Allocates log buffer 
	* 2) sets curr pointer position 
	*	 at the start of the lobuffer,
	* 3) Opens log File
	* @param log_buffer_size_in_bytes
	*/
	void init_profiler(unsigned log_buffer_size_in_bytes);


	/**
	* Getter for log buffer.
	* @return 
	*/
	char* get_log_buffer();

	/**
	* Returns time.
	*
	* @return 
	*/
	time_t get_time_stamp();

	/**
	* Close log file.
	*
	* @return 
	*/
	bool close_log_file();

	/**
	* Opens log file. 	
	* @return 
	*/
	bool open_log_file();

	/**
	* Update log file when buffer is full 
	* @return 
	*/
	bool update_log_file();

	/**
	* Creates an entry to  buffer with the accelerator list
	* of specific type.  
	* @param type
	*/
	void log_vine_accel_list(vine_accel *** accels);

	//void log_vine_accel_location(vine_accel * accel);

	//void log_vine_accel_type(vine_accel * accel);

	//void log_vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat);

	/**
	* Creates an entry to  buffer with the accelerator that 
	* has been acquired. 
	* @param accel
	*/
	void log_vine_accel_acquire(vine_accel * accel);

	/**
	* Creates an entry to  buffer with the accelerator that 
	* has been released. 
	* @param accel
	*/
	void log_vine_accel_release(vine_accel * accel);

	/**
	* Creates an entry to  buffer with the proccess 
	* that has been registered.
	* @param type
	* @param func_name
	* @param func_bytes
	* @param func_bytes_size
	*/
	void log_vine_proc_register(vine_accel_type_e type,const char * func_name,
					const void * func_bytes,size_t func_bytes_size,vine_proc* ret_val);

	/**
	* Creates an entry to  buffer with the proccess 
	* that has been retrieved.
	* @param type
	* @param func_name
	*/
	void log_vine_proc_get(vine_accel_type_e type,const char * func_name,void* return_value);

	/**
	* Creates an entry to  buffer with the proccess that 
	* has been deleted.
	* @param func
	*/
	void log_vine_proc_put(vine_proc * func);

	//void log_vine_data_alloc(size_t size,vine_data_alloc_place_e place);

	void  log_vine_data_deref(vine_data * data);

	void log_vine_data_free(vine_data * data);

	/**
	* Creates an entry to  buffer with the task that 
	* has been issued.
	*
	* @param accel
	* @param proc
	* @param input
	* @param output
	*/
	void log_vine_task_issue(vine_accel * accel,vine_proc * proc,vine_data ** input,vine_data ** output);

	void log_vine_task_stat(vine_task * task,vine_task_stats_s * stats);

	/**
	* Creates an entry to  buffer that writes 
	* we wait for task completion.
	* @param task
	*/
	void log_vine_task_wait_start(vine_task * task);

	/**
	* Creates an entry to  buffer that writes 
	* task success or failed.
	* @param task
	*/
	void log_vine_task_wait_end(vine_task * task);

	
#endif
