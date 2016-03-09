#include "vine_talk.h"
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

int flag = 0;
int vine_accel_list(vine_accel_type_e type,vine_accel *** accels)
{

}

vine_accel_loc_s vine_accel_location(vine_accel * accel)
{

}

vine_accel_type_e vine_accel_type(vine_accel * accel)
{

}

vine_accel_state_e vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat)
{

}

int vine_accel_acquire(vine_accel * accel)
{

}

void vine_accel_release(vine_accel * accel)
{

}

vine_proc * vine_proc_register(vine_accel_type_e type,const char * func_name,const void * func_bytes,size_t func_bytes_size)
{

}

vine_proc * vine_proc_get(vine_accel_type_e type,const char * func_name)
{
	if( !is_initialized  )
	{	
		unsigned log_buffer_size = 200;
		init_profiler(log_buffer_size);
		flag = 1;
	}
	vine_proc* ret_proc;
	log_vine_proc_get(type,func_name,ret_proc);
}

int vine_proc_put(vine_proc * func)
{

}

vine_data * vine_data_alloc(size_t size,vine_data_alloc_place_e place)
{

}

size_t vine_data_size(vine_data * data)
{

}

void * vine_data_deref(vine_data * data)
{

}

void vine_data_free(vine_data * data)
{

}

vine_task * vine_task_issue(vine_accel * accel,vine_proc * proc,vine_data ** input,vine_data ** output)
{

}

vine_task_state_e vine_task_stat(vine_task * task,vine_task_stats_s * stats)
{

}

vine_task_state_e vine_task_wait(vine_task * task)
{

}
//////////////////////////////////////////////////////////////////////////////

void init_log_buffer(unsigned  bytes)
{
	log_buffer_start_ptr = (char*) malloc(bytes);
	//printf ("log_buffer allocates %d bytes \n",bytes);
}




void init_profiler(unsigned log_buffer_size_in_bytes)
{
	
	init_log_buffer(log_buffer_size_in_bytes);
	curr_entry = log_buffer_start_ptr;
	open_log_file();
	log_buffer_size = log_buffer_size_in_bytes;
	is_initialized  = TRUE;
}

char* get_log_file_name()
{
	char hostname[1024];
	char buffer[30];
	struct timeval tv;
	time_t curtime;
	char fileName[100];


	hostname[1023] = '\0';
	gethostname(hostname, 1023);

	gettimeofday(&tv, NULL); 
	curtime=tv.tv_sec;

	strftime(buffer,30,"%m-%d-%Y-%T",localtime(&curtime));	
	sprintf(fileName,"trace_%s_%d_%s.scv",hostname,getpid(),buffer);

	return strdup(fileName);
}


bool open_log_file(){
	log_file = open(get_log_file_name(),O_CREAT|O_RDWR);
	if(  log_file < 0 ){
		perror("Error fopen failed:");	
		return 0;
	}
	return 1;

}

bool close_log_file(){
}

bool update_log_file(){
	//printf("update log file\n");
	if( (write(log_file,log_buffer_start_ptr,strlen(log_buffer_start_ptr)+1) )  != ( strlen(log_buffer_start_ptr)+1) ){
		perror("Update log file failed\n");
	}
	fsync(log_file);
	return 0;
}
void log_vine_accel_list(vine_accel *** accels){

}

void log_vine_accel_location(vine_accel * accel){

}

void log_vine_accel_type(vine_accel * accel){

}

void log_vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat){

}

void log_vine_accel_acquire(vine_accel * accel){

}

void log_vine_accel_release(vine_accel * accel){
}

void log_vine_proc_register(vine_accel_type_e type,const char * func_name,
						const void * func_bytes,size_t func_bytes_size,vine_proc* ret_val){
}

char* enum_vine_accel_type_to_str(vine_accel_type_e type)
{
	char* array[] = {"ANY","GPU","GPU_SOFT","CPU","FPGA"};
	return 	array[type];
}

bool has_buffer_enough_space(int new_entry_bytes)
{
	char* end_buffer_ptr			= log_buffer_start_ptr +(sizeof(char) * log_buffer_size);
	char* next_entry_end_ptr		= curr_entry + (sizeof(char)*new_entry_bytes); 

	//printf("%p ? %p \n",end_buffer_ptr,next_entry_end_ptr);
	//printf("%d \n",( ( next_entry_end_ptr <=  end_buffer_ptr) ? 1 : 0 ));

	return  ( ( next_entry_end_ptr <=  end_buffer_ptr) ? 1 : 0 );
}
void update_curr_entry(int size_of_new_entry)
{
	/*atomic update of current position pointer*/
	__sync_fetch_and_add(&curr_entry, (size_of_new_entry*sizeof(char) ));
}
int create_log_entry(char* func_id,char* ret_status,int task_duration,
				vine_accel_type_e accel_type,unsigned incnt,unsigned outcnt,
				char** indata,char** outdata,char** new_entry_buffer)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	size_t		timestamp	= tv.tv_sec*1000000+tv.tv_usec; //fix
	int			core_id		= sched_getcpu(); 
	int			thread_id	= getpid();
	char * accel_type_str = 	enum_vine_accel_type_to_str(accel_type);
	int new_entry_size = sprintf(*new_entry_buffer,"%zu,%d,%d,%p,%s,%d,%s,%d,%d\n",
					timestamp,core_id,thread_id,func_id,ret_status,task_duration,accel_type_str,incnt,outcnt);
	return new_entry_size;

}
bool update_profiler(char* new_entry,unsigned new_entry_size){
	//printf("new_entry_size = %d\n",(new_entry_size+1));

	if(has_buffer_enough_space(new_entry_size))
	{	
		//printf("curr_entry addr before update: %p \n",curr_entry);
		char* copy_position = curr_entry;
		/*atomic update of current position pointer*/
		update_curr_entry(new_entry_size);
		//printf("curr_entry addr  after update: %p \n",curr_entry);
		//printf("copy_posirion addr  after update: %p \n",copy_position);
		memcpy(copy_position,new_entry,(new_entry_size+1));	
	}
	else
	{
		//printf("write to log file\n");
		update_log_file();
		curr_entry = log_buffer_start_ptr;	/*reset pointer at the start of log buffer*/
		memcpy(curr_entry,new_entry,strlen(new_entry)+1);	
	}
	//printf("log_buffer_start_ptr addr: %p \n",log_buffer_start_ptr);

	printf("%s \n",log_buffer_start_ptr);


}
void log_vine_proc_get(vine_accel_type_e type,const char * func_name,void* return_val)
{
	/*CREATE ENTRY*/
	char* new_entry = malloc(300*sizeof(char));
	int new_entry_size = create_log_entry(return_val,0,0,type,0,0,0,0,&new_entry);
	
	/*UPDATE PROFILER*/
	update_profiler(new_entry,new_entry_size);
	free(new_entry);
}

void  log_vine_proc_put(vine_proc * func){

}

void log_vine_data_alloc(size_t size,vine_data_alloc_place_e place){

}

void  log_vine_data_deref(vine_data * data)
{


}

void log_vine_data_free(vine_data * data)
{
}

void log_vine_task_issue(vine_accel * accel,vine_proc * proc,vine_data ** input,vine_data ** output)
{
}

void log_vine_task_stat(vine_task * task,vine_task_stats_s * stats)
{
}

void log_vine_task_wait(vine_task * task)
{
}


