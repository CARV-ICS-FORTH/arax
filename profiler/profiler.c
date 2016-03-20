#include "profiler.h"
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
#include <pthread.h>
#include <fcntl.h>
#include <stdio.h>
#include <math.h>


void init_profiler()
{

	log_buffer_size   = get_log_buffer_size();
	printf("%d\n",log_buffer_size);
	log_buffer_start_ptr =  malloc(log_buffer_size);
	curr_entry_pos = -1;
	open_log_file();
	is_initialized  = TRUE;
	log_buffer_is_full = FALSE;

}
int get_log_buffer_size(){

	char buf[100];
	memset(buf,0,100);
	size_t nbytes= sizeof(buf);
	ssize_t bytes_read;
	int buffer_size = 0;
	int conf_fd= open(CONFIG_NAME,O_RDONLY);
	char* ptr;
	if(conf_fd< 0)
	{
		fprintf(stderr,"PROFILER: open syscall failed to open %s",CONFIG_NAME);
		perror(" ");
		exit(-1);
	}
	char c ;
	int i = 0;
	do{
		bytes_read = read(conf_fd, &c, 1);
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
	sprintf(fileName,"trace_%s_%d_%s.scv",hostname,getpid(),buffer);

	return strdup(fileName);
}

void open_log_file(){
	char* fileName = get_log_file_name();
	log_file = open(fileName,O_CREAT|O_RDWR);/*global var*/
	if(log_file < 0)
	{
		perror("PROFILER: open syscall failed ");
		exit(-1);
	}
	free(fileName);

}

void close_profiler()
{
	//add /0 to the end of buffer

}
void close_log_file(){
}

bool update_log_file(){
	//printf("update log file\n");
	/*unsigned bytes = (curr_entry - log_buffer_start_ptr);
	if( (write(log_file,log_buffer_start_ptr,bytes )  != bytes) ){
		perror("Update log file failed\n");
	}
	//IMPORTANT FIX WIPE log_buffer
	fsync(log_file);*/
	return 0;
}

void log_vine_accel_list(vine_accel_type_e type,vine_accel *** accels,const char* func_id,int task_duration,void* return_value)
{
	log_entry* entry;
	entry		=get_log_buffer_ptr();

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
	entry		=get_log_buffer_ptr();

	init_log_entry(entry);

	entry->accel			= accel;
	entry->accel_stat		= stat;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= return_val;

}
vine_accel_loc_s log_vine_accel_location(vine_accel * accel,const char* func_id,
						vine_accel_loc_s return_val,int task_duration)
{
	log_entry* entry;
	entry		= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_val;


}

void log_vine_accel_type(vine_accel * accel,const char* func_id,int task_duration,void* return_value)
{
	log_entry* entry;
	entry		= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= return_value;

}


void log_vine_task_stat(vine_task * task,vine_task_stats_s * stats,const char* func_id,int task_duration,vine_task_state_e return_value)
{
	log_entry* entry;
	entry		= get_log_buffer_ptr();

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
	entry		= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->accel			= accel;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_val;

}

void log_vine_accel_release(vine_accel * accel,const char* func_id,vine_accel_loc_s return_val,int task_duration)
{

	log_entry* entry;
	entry		= get_log_buffer_ptr();

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

inline char* enum_vine_accel_type_to_str(vine_accel_type_e type)
{
	if(type < 0 ) return NULL;
	char* array[] = {"ANY","GPU","GPU_SOFT","CPU","FPGA"};
	return 	array[type];
}

bool has_buffer_enough_space()
{
	int total_log_entries = log_buffer_size/sizeof(log_entry);
	return ((curr_entry_pos < total_log_entries)?1:0);

}

void debug_print_log_entry(log_entry* entry){
	printf("%zu,%d,%d,%s,%zu,%p",
				entry-> timestamp,
				entry-> core_id,
				entry-> proccess_id,
				entry-> func_id,
				entry-> task_duration,
				entry-> return_value);
	if(entry-> accels) printf(",%p",entry->accels);

	if(entry->accel) printf(",%p",	entry-> accel);
	if(entry->accel_stat) printf(",%p",entry->accel_stat);
	if(entry-> accel_type != -1) printf(",%d",entry->accel_type);
	if(entry-> func_name) printf(",%p",entry-> func_name);
	if(entry-> func_bytes) printf(",%p",entry->func_bytes);
	if(entry-> func_bytes_size) printf(",%zu",entry-> func_bytes_size);
	if(entry->func)				printf(",%p",entry->func);

	if(entry-> accel_place != -1) printf(",%d",entry->accel_place);

	if(entry-> data) printf(",%p:%zu",entry->data,vine_data_size(entry->data));
	if(entry-> in_data) printf(",%p",entry->in_data);
	if(entry-> out_data) printf(",%p",entry->out_data);
	if(entry-> task) printf(",%p",entry->task);
	if(entry-> task_stats) printf(",%p",entry->task_stats);
	printf("\n");

}
void debug_print_log_buffer(){
	int i;
	for(i = 0; i <= curr_entry_pos; i++){
		debug_print_log_entry(&log_buffer_start_ptr[i]);
	}

}

bool update_profiler(log_entry* new_entry){
	if( has_buffer_enough_space() )
	{
		__sync_fetch_and_add(&curr_entry_pos, 1);
		memcpy( &log_buffer_start_ptr[curr_entry_pos], new_entry, sizeof(log_entry));
	}
	else
	{

		update_log_file();
		curr_entry_pos = 0;
		memcpy(&log_buffer_start_ptr[curr_entry_pos],new_entry,sizeof(new_entry));
	}


}
/*
bool update_profiler( char* new_entry,unsigned new_entry_size){
	//printf("new_entry_size = %d\n",(new_entry_size+1));

	if(has_buffer_enough_space(new_entry_size))
	{
		char* copy_position;
		copy_position = curr_entry;

		update_curr_entry(new_entry_size);


		strcat(log_buffer_start_ptr,new_entry);
	//	memcpy(copy_position,new_entry,strlen(new_entry));
		printf("new entry size is %zu\n",strlen(new_entry));
		printf("buffer size is %zu\n",strlen(log_buffer_start_ptr));
		printf("LOG BUFFER \n");
		printf("%s\n\n\n\n",log_buffer_start_ptr);
		//printf("%s\n",copy_position);

	}
	else
	{
		printf("write to log file\n");
		update_log_file();
		curr_entry = log_buffer_start_ptr;
		memcpy(curr_entry,new_entry,strlen(new_entry)+1);
	}
	//printf("log_buffer_start_ptr addr: %p \n",log_buffer_start_ptr);
	//printf("%s \n",log_buffer_start_ptr);


}
*/

/*
void logger( const char* func_id,int task_duration,char* log_msg)
{

	char *new_entry;
	int new_entry_prefix_length;
	int new_entry_length;

	new_entry = malloc(2048 *sizeof(char));

	new_entry_prefix_length = create_log_entry_prefix(func_id,task_duration,&new_entry);


	new_entry_length = new_entry_prefix_length+strlen(log_msg);
	assert(new_entry_length < 2048);
	strcat(new_entry,log_msg);

	update_profiler(new_entry,new_entry_length);

	free(new_entry);


}*/
void init_log_entry(log_entry* entry){

	memset(entry,0,sizeof(log_entry));
	entry->accel_type	= -1;
	entry->accel_place	= -1;

	int prefix_size;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	entry->timestamp			= tv.tv_sec*1000000+tv.tv_usec;
	entry->core_id				= sched_getcpu();
	entry->proccess_id			= getpid();


}

log_entry* get_log_buffer_ptr(){
	while(log_buffer_is_full);

	curr_entry_pos++;
	return &(log_buffer_start_ptr[curr_entry_pos]);
}

void log_vine_proc_get(vine_accel_type_e type,const char * func_name,const char* func_id,int task_duration,vine_proc* return_val)
{
	log_entry* entry;
	entry	= get_log_buffer_ptr();

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
	entry	= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->func				= func;
	entry->func_id			= func_id;
	entry->task_duration	= task_duration;
	entry->return_value		= &return_value;
	debug_print_log_buffer();



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
	entry	= get_log_buffer_ptr();

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
	entry	= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->data				= data;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= return_value;


}

void log_vine_data_free(vine_data * data,const char* func_id,int task_duration)
{
	log_entry* entry;
	entry	= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->data				= data;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;

}

void log_vine_task_issue(vine_accel * accel,
						vine_proc * proc,
						vine_data ** input,
						vine_data ** output,
						const char* func_id,
						int task_duration ,
						vine_task* return_value)
{
	log_entry* entry;
	entry	= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->accel			= accel;
	entry->func				= proc;
	entry->in_data			= input;
	entry->out_data			= output;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= return_value;




}


void log_vine_task_wait(vine_task * task,const char* func_id,int task_duration,vine_task_state_e return_value)
{
	log_entry* entry;
	entry	= get_log_buffer_ptr();

	init_log_entry(entry);

	entry->task				= task;
	entry->task_duration	= task_duration;
	entry->func_id			= func_id;
	entry->return_value		= &return_value;

}
