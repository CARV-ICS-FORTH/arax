#include "vine_talk.h"
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <stdarg.h>


static FILE* get_log_file_ptr(){
	if(log_file == NULL){
		if( (log_file = fopen("./log.txt","w+")) == NULL){
			perror("Error fopen failed:");	
			return NULL;
		}
	}	
	return log_file;
}

bool close_log_file(){
	if( fclose(get_log_file_ptr()) == EOF){
		perror("Error fclose failed:");
		return -1;
	}
	return 0;
}

bool update_log_file(){
	FILE* file = get_log_file_ptr();
	assert(file != NULL);


	
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

void log_vine_proc_register(vine_accel_type_e type,const char * func_name,const void * func_bytes,size_t func_bytes_size){
}

void log_vine_proc_get(vine_accel_type_e type,const char * func_name){

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


