#define _GNU_SOURCE
#include "trace.h"
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <math.h>
#include <linux/limits.h>
#include <sched.h>


__attribute__ ((__constructor__))
  void profiler_constructor(void) {
	init_profiler();
}

__attribute__((__destructor__))
    void profiler_destructor(void){
	close_profiler();
	pthread_mutex_destroy(&lock);
}


void init_profiler()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);

	if (pthread_mutex_init(&lock, NULL) != 0)
	{
		fprintf(stderr,"PROFILER: mutex init failed\n");
		exit(-1);
	}

	log_buffer_size   = get_log_buffer_size();
	log_buffer_start_ptr =  malloc(log_buffer_size);
	curr_entry_pos = -1;
	open_log_file();
	log_buffer_is_full = 0;
	start_of_time = tv.tv_sec*1000000+tv.tv_usec;

}
int get_log_buffer_size(){

	char buf[100];
	memset(buf,0,100);
	int buffer_size = 0;
	char *home_path = getenv("HOME");
	int home_path_length = strlen(home_path);
	char config_path[PATH_MAX];

	assert((home_path_length + strlen(CONFIG_NAME) + 1) < PATH_MAX);

	if (strcpy(config_path, home_path) == NULL)
		fprintf(stderr, "PROFILER: failed to copy string at %s:%d\n", __FILE__, __LINE__);

	config_path[home_path_length] = '/';
	config_path[home_path_length+1] = '\0';

	if (strcat(config_path, CONFIG_NAME) == NULL)
		fprintf(stderr, "PROFILER: failed to concat string at %s:%d\n", __FILE__, __LINE__);

	int conf_fd = open(config_path,O_RDONLY);
	char* ptr;
	if(conf_fd< 0)
	{
		fprintf(stderr,"PROFILER: open syscall failed to open \"%s\"",config_path);
		perror(" ");
		exit(-1);
	}
	char c ;
	int i = 0;
	do{
		read(conf_fd, &c, 1);
		buf[i] = c;
		i++;
	}while(c != '\n');
	assert(i <= 100);

	ptr = strstr(buf," ");
	buffer_size = atoi(ptr);
	assert(buffer_size > 0);
	close(conf_fd);

	return buffer_size;
}

char* get_log_file_name()
{
	char hostname[1024];
	char buffer[30];
	struct timeval tv;
	time_t curtime;
	char fileName[100];

	/*after log file is created ,
	*we must not call this function*/	
	///assert(log_file == 0);

	hostname[1023] = '\0';
	gethostname(hostname, 1023);

	gettimeofday(&tv, NULL); 
	curtime=tv.tv_sec;

	strftime(buffer,30,"%m-%d-%Y-%T",localtime(&curtime));	
	sprintf(fileName,"trace_%s_%d_%s.csv",hostname,getpid(),buffer);

	return strdup(fileName);
}

void open_log_file(){
	char* fileName = get_log_file_name();
	log_file = open(fileName,O_CREAT|O_RDWR,0777);/*global var*/
	if(log_file < 0)
	{
		perror("PROFILER: open syscall failed ");
		exit(-1);
	}	
	free(fileName);
	dprintf(log_file,"%s\n\n","Timestamp,Core Id,Thread Id,Function Id,Task Duration,Return Value");


}

void close_profiler()
{
	update_log_file();
	free(log_buffer_start_ptr);
	close(log_file);	
}

void update_log_file(){
	print_log_buffer_to_fd(log_file);
	fsync(log_file);
}

void log_vine_accel_list(vine_accel_type_e type,vine_accel *** accels,const char* func_id,int task_duration,void* return_value)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel_type		= type;
	entry->accels			= accels;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= return_value;

}

void log_vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat,const char* func_id,int task_duration,void* return_val)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel			= accel;
	entry->accel_stat		= stat;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= return_val;

}
void log_vine_accel_location(vine_accel * accel,const char* func_id,
						vine_accel_loc_s return_val,int task_duration)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_val;


}

void log_vine_accel_type(vine_accel * accel,const char* func_id,int task_duration,void* return_value)
{
	log_entry* entry;

	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= return_value;

}


void log_vine_task_stat(vine_task * task,vine_task_stats_s * stats,const char* func_id,int task_duration,vine_task_state_e return_value)
{
	log_entry* entry;

	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->task				= task;
	entry->task_stats		= stats;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_value;
}

void log_vine_accel_acquire(vine_accel * accel,const char* func_id,vine_accel_loc_s return_val,int task_duration)
{

	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_val;

}

void log_vine_accel_release(vine_accel * accel,const char* func_id,vine_accel_loc_s return_val,int task_duration)
{

	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_val;

}

void log_vine_proc_register(vine_accel_type_e type,const char * proc_name,
						const void * func_bytes,size_t func_bytes_size,const char* func_id,
						int task_duration,void* return_value)
{

}

unsigned int is_log_buffer_full()
{
	int total_log_entries = log_buffer_size/sizeof(log_entry);
	
	return ((curr_entry_pos >= (total_log_entries-1))?1:0);
	
}

void print_log_entry_to_fd(int fd,log_entry* entry){
	dprintf(fd,"%zu,%d,%lu,%s,%zu,%p",
				entry-> timestamp,
				entry-> core_id,
				entry->thread_id ,
				entry-> func_id,
				entry-> task_duration,
				entry-> return_value);
	if(entry-> accels) dprintf(fd,",%p",entry->accels);

	if(entry->accel)				dprintf(fd,",%p",entry-> accel);
	if(entry->accel_stat)			dprintf(fd,",%p",entry->accel_stat);
	if(entry-> accel_type != -1)	dprintf(fd,",%d",entry->accel_type);
	if(entry-> func_name)			dprintf(fd,",%p",entry-> func_name);
	if(entry-> func_bytes)			dprintf(fd,",%p",entry->func_bytes);
	if(entry-> func_bytes_size)		dprintf(fd,",%zu",entry-> func_bytes_size);
	if(entry->func)					dprintf(fd,",%p",entry->func);

	if(entry-> accel_place != -1)	dprintf(fd,",%d",entry->accel_place);

	if(entry-> data)				dprintf(fd,",%p:%zu",entry->data,vine_data_size(entry->data));
	if(entry->args)					dprintf(fd,",%p",entry->args);
	if(entry-> in_data)				dprintf(fd,",%p",entry->in_data);
	if(entry-> out_data)			dprintf(fd,",%p",entry->out_data);
	if(entry-> task)				dprintf(fd,",%p",entry->task);
	if(entry-> task_stats)			dprintf(fd,",%p",entry->task_stats);
	dprintf(fd,"\n");

}
void print_log_buffer_to_fd(){
	int i;
	for(i = 0; i <= curr_entry_pos; i++){
		print_log_entry_to_fd(log_file,&log_buffer_start_ptr[i]);
	}

}
void debug_print_log_entry(FILE* fstream,log_entry* entry){
	fprintf(fstream,"%zu,%d,%lu,%s,%zu,%p",
				entry-> timestamp,
				entry-> core_id,
				entry-> thread_id,
				entry-> func_id,
				entry-> task_duration,
				entry-> return_value);
	if(entry-> accels) printf(",%p",entry->accels);

	if(entry->accel)				fprintf(fstream,",%p",	entry-> accel);
	if(entry->accel_stat)			fprintf(fstream,",%p",entry->accel_stat);
	if(entry-> accel_type != -1)	fprintf(fstream,",%d",entry->accel_type);
	if(entry-> func_name)			fprintf(fstream,",%p",entry-> func_name);
	if(entry-> func_bytes)			fprintf(fstream,",%p",entry->func_bytes);
	if(entry-> func_bytes_size)		fprintf(fstream,",%zu",entry-> func_bytes_size);
	if(entry->func)					fprintf(fstream,",%p",entry->func);

	if(entry-> accel_place != -1)	fprintf(fstream,",%d",entry->accel_place);

	if(entry-> data)				fprintf(fstream,",%p:%zu",entry->data,vine_data_size(entry->data));
	if(entry-> in_data)				fprintf(fstream,",%p",entry->in_data);
	if(entry-> out_data)			fprintf(fstream,",%p",entry->out_data);
	if(entry-> task)				fprintf(fstream,",%p",entry->task);
	if(entry-> task_stats)			fprintf(fstream,",%p",entry->task_stats);
	fprintf(fstream,"\n");

}		
void debug_print_log_buffer(FILE* file){
	int i;
	for(i = 0; i <= curr_entry_pos; i++){
		debug_print_log_entry(file,&log_buffer_start_ptr[i]);
	}

}
void init_log_entry(log_entry* entry){
	
	memset(entry,0,sizeof(log_entry));
	entry->accel_type	= -1;
	entry->accel_place	= -1;
	
	struct timeval tv;
	gettimeofday(&tv, NULL);
	entry->timestamp			= (tv.tv_sec*1000000+tv.tv_usec) - start_of_time; 
	entry->core_id				= sched_getcpu(); 
	entry->thread_id			= pthread_self();
	

}

log_entry* get_log_buffer_ptr(){

	if(is_log_buffer_full()){
		update_log_file();
		memset(log_buffer_start_ptr,0,log_buffer_size);
		curr_entry_pos = -1; 
	}
	curr_entry_pos++;

	return &(log_buffer_start_ptr[curr_entry_pos]);   
}

void log_vine_proc_get(vine_accel_type_e type,const char * func_name,const char* func_id,int task_duration,vine_proc* return_val)
{
	log_entry* entry;

	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);

	init_log_entry(entry);

	entry->accel_type		= type;
	entry->func_name		= func_name;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= return_val;

}

void log_vine_proc_put(vine_proc * func,const char* func_id,int task_duration,int return_value)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->func				= func;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_value;



}

inline char* enum_vine_data_alloc_place_to_str(vine_data_alloc_place_e place)
{
	if( place < 0) return NULL;
	char* array[] = {"HostOnly","AccelOnly","Both"};
	return 	array[place-1];
}
	
void log_vine_data_alloc(size_t size,vine_data_alloc_place_e place,
						int task_duration,const char* func_id,void* return_val)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);

	init_log_entry(entry);

	entry->data_size		= size;
	entry->accel_place		= place;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= return_val;


}

void  log_vine_data_deref(vine_data * data ,const char* func_id,int task_duration,void* return_value)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->data				= data;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= return_value;


}

void log_vine_data_free(vine_data * data,const char* func_id,int task_duration)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->data				= data;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;

}

void log_vine_task_issue(vine_accel * accel,
						vine_proc * proc,	
						vine_data*	args,
						vine_data ** input,
						vine_data ** output,
						const char* func_id,
						int task_duration ,
						vine_task* return_value)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->accel			= accel;
	entry->func				= proc;
	entry->args				= args;
	entry->in_data			= input;
	entry->out_data			= output;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= return_value;
}


void log_vine_task_wait(vine_task * task,const char* func_id,int task_duration,vine_task_state_e return_value)
{
	log_entry* entry;
	pthread_mutex_lock(&lock);
	entry	= get_log_buffer_ptr();
	pthread_mutex_unlock(&lock);


	init_log_entry(entry);

	entry->task				= task;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= &return_value;

}


