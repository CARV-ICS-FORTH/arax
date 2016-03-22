#ifndef PROFILER_H
	#define PROFILER_H
	#define		TRUE		1
	#define		FALSE		0
	#define		CONFIG_NAME	"/kavros/test/vine_talk/profiler/vine_profiler.conf"
	#include "vine_talk.h"
	#include <unistd.h>
	#include <pthread.h>

	typedef struct Entry{
		size_t					timestamp;
		int						core_id;
		pthread_t				thread_id;
		const char*				func_id;
		size_t					task_duration;
		void*					return_value;
		vine_accel***			accels;
		vine_accel*				accel;
		vine_accel_stats_s*		accel_stat;
		vine_accel_type_e		accel_type;
		const char*				func_name;
		const void*				func_bytes;
		size_t					func_bytes_size;
		vine_proc*				func;
		vine_data_alloc_place_e accel_place;
		vine_data*				data;
		size_t					data_size;
		vine_data**				in_data;
		vine_data**				out_data;
		vine_task*				task;
		vine_task_stats_s*		task_stats;
	}log_entry;


	/**
	* useful for return value.
	*/
	typedef unsigned int bool;
	int	curr_entry_pos;
	int				log_buffer_size;
	log_entry*		log_buffer_start_ptr;
	int				log_file;
	bool			is_initialized;
	bool			log_buffer_is_full ;
	pthread_mutex_t lock;

/*
	int create_log_entry(char* func_id,char* ret_status,int task_duration,
					vine_accel_type_e accel_type,vine_data_alloc_place_e place,
					unsigned incnt,unsigned outcnt,
					char** indata,char** outdata,char** new_entry_buffer);
*/
	log_entry* get_log_buffer_ptr();
	void create_log_entry(log_entry* entry);
	void init_log_entry(log_entry* entry);


	int get_log_buffer_size();

	bool is_log_buffer_full();

	//char* get_next_entry_ptr(int size_of_new_entry);
	//void create_new_entry(char* func_id


	/**
	* Returns accel type as a string
	*/
	char* enum_vine_accel_type_to_str(vine_accel_type_e type);
	char* enum_vine_data_alloc_place_to_str(vine_data_alloc_place_e place);


	char* enum_vine_accel_type_to_str(vine_accel_type_e type);

	/**
	* Returns the name of log buffer
	* which is in the form of this
	*  < trace_hostname_pid_date.csv >
	* @return  log file name
	*/
	char* get_log_file_name();



	/**
	* 1) Allocates log buffer
	* 2) sets curr pointer position
	*	 at the start of the lobuffer,
	* 3) Opens log File
	* @param log_buffer_size_in_bytes
	*/
	void init_profiler();


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
	void close_log_file();

	/**
	* Opens log file.
	* @return
	*/
	void open_log_file();

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
	void log_vine_accel_list(vine_accel_type_e type,vine_accel *** accels,const char* func_id,
						int task_duration,void* return_value);

	vine_accel_loc_s log_vine_accel_location(vine_accel * accel,const char* func_id,vine_accel_loc_s return_val,int task_duration);

	void log_vine_accel_type(vine_accel * accel,const char* func_id,int task_duration,void* return_value);

	void log_vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat,const char* func_id,int task_duration,void* return_value);

	/**
	* Creates an entry to  buffer with the accelerator that
	* has been acquired.
	* @param accel
	*/
	void log_vine_accel_acquire(vine_accel * accel,const char* func_id,vine_accel_loc_s return_val,int task_duration);

	/**
	* Creates an entry to  buffer with the accelerator that
	* has been released.
	* @param accel
	*/
	void log_vine_accel_release(vine_accel * accel,const char* func_id,vine_accel_loc_s return_val,int task_duration);

	void log_vine_proc_register(vine_accel_type_e type,const char * proc_name,
						const void * func_bytes,size_t func_bytes_size,const char* func_id,
						int task_duration,void* return_value);

	/**
	* Creates an entry to  buffer with the proccess
	* that has been retrieved.
	* @param type
	* @param func_name
	*/
	void log_vine_proc_get(vine_accel_type_e type,const char * func_name,const char* func_id,int task_duration,vine_proc* return_value);


	void log_vine_proc_put(vine_proc * func,const char* func_id,int task_duration,int return_value);

	void log_vine_data_alloc(size_t size,vine_data_alloc_place_e place,
						int task_duration,const char* func_id,vine_data* return_val);


	void  log_vine_data_deref(vine_data * data,const char* func_id,int task_duration,void* return_value);

	void log_vine_data_free(vine_data * data,const char* func_id,
							int task_duration);

	void log_vine_task_issue(vine_accel * accel,
							vine_proc * proc,
							vine_data ** input,
							vine_data ** output,
							const char* func_id,
							int task_duration,
							vine_task* return_value);

	void log_vine_task_stat(vine_task * task,vine_task_stats_s * stats,
							const char* func_id,int task_duration,vine_task_state_e return_value);

	/**
	* Creates an entry to  buffer that writes
	* we wait for task completion.
	* @param task
	*/
	void log_vine_task_wait(vine_task * task,const char* func_id ,
						int task_duration,vine_task_state_e return_value);

	/*void logger( const char* func_id,int task_duration,char* log_msg);*/
	bool update_profiler(log_entry*  entry);
	
	void debug_print_log_buffer(FILE*);
	void debug_print_log_entry(FILE*,log_entry* entry);

	void print_log_buffer();
	void print_log_entry_to_fd(int fd,log_entry* entry);
	void close_profiler();
#endif
