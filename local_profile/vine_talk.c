#include <vine_talk.h>
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


int vine_accel_list(vine_accel_type_e type,vine_accel *** accels)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	int task_duration=0;
	int  return_value;
	log_vine_accel_list(type,accels,__FUNCTION__,task_duration,&return_value);
}

vine_accel_loc_s vine_accel_location(vine_accel * accel)
{
	if( !is_initialized  )
	{
		init_profiler();
	}

	vine_accel_loc_s return_val;
	int task_duration=0;
	log_vine_accel_location( accel, __FUNCTION__,return_val,task_duration);
}

vine_accel_type_e vine_accel_type(vine_accel * accel)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	int task_duration = 0;
	vine_accel_type_e return_value;
	log_vine_accel_type(accel,__FUNCTION__,return_value,&task_duration);


}

vine_accel_state_e vine_accel_stat(vine_accel * accel,vine_accel_stats_s * stat)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	int task_duration = 0;
	vine_accel_state_e return_value;
	log_vine_accel_stat(accel,stat,__FUNCTION__,task_duration,&return_value);
}

int vine_accel_acquire(vine_accel * accel)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	vine_accel_loc_s return_value;
	int task_duration = 0;
	log_vine_accel_acquire(accel,__FUNCTION__,return_value,task_duration);
}

void vine_accel_release(vine_accel * accel)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	int task_duration = 0;
	vine_accel_loc_s return_value;
	log_vine_accel_release(accel,__FUNCTION__,return_value,task_duration);
}

vine_proc * vine_proc_register(vine_accel_type_e type,const char * func_name,const void * func_bytes,size_t func_bytes_size)
{
	if( !is_initialized  )
	{
		init_profiler();
	}

	int task_duration = 0;
	vine_proc* return_value;

	log_vine_proc_register(type,func_name,
						func_bytes,func_bytes_size,__FUNCTION__,
						task_duration,return_value);

}

vine_proc * vine_proc_get(vine_accel_type_e type,const char * proc_name)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	vine_proc* ret_proc;
	int task_duration = 0;
	log_vine_proc_get(type,proc_name,__FUNCTION__,task_duration,ret_proc);
	return ret_proc;

}

int vine_proc_put(vine_proc * func)
{
	if( !is_initialized  )
	{
		init_profiler();
	}



	int return_value;
	int task_duration = 0;
	log_vine_proc_put(func,__FUNCTION__, task_duration,return_value);
}

vine_data * vine_data_alloc(size_t size,vine_data_alloc_place_e place)
{
	if( !is_initialized  )
	{
		init_profiler();
	}


	int task_duration=0;
	vine_data* return_val;
	log_vine_data_alloc(size,place,task_duration,__FUNCTION__,return_val);

}

size_t vine_data_size(vine_data * data)
{

	return 0;

}

void * vine_data_deref(vine_data * data)
{
	if( !is_initialized  )
	{
		init_profiler();
	}


	int task_duration=0;
	void* return_val;

	log_vine_data_deref(data,__FUNCTION__,task_duration,return_val);


}

void vine_data_free(vine_data * data)
{
	if( !is_initialized  )
	{
		init_profiler();
	}

	int task_duration=0;
	log_vine_data_free( data,__FUNCTION__,task_duration);


}
vine_task* vine_task_issue(vine_accel *accel, vine_proc *proc, vine_data *args,
                           vine_data **input, vine_data **output)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	int task_duration=0;
	vine_task* return_value;
	log_vine_task_issue(accel, proc, input,output,__FUNCTION__,task_duration,return_value);
	return return_value;
}

vine_task_state_e vine_task_stat(vine_task * task,vine_task_stats_s * stats)
{
	if( !is_initialized  )
	{
		init_profiler();
	}
	int task_duration=0;
	vine_task_state_e  return_value;

	log_vine_task_stat(task,stats,__FUNCTION__,task_duration,return_value);
}

vine_task_state_e vine_task_wait(vine_task * task)
{
	if( !is_initialized  )
	{
		init_profiler();
	}

	int task_duration=0;
	vine_task_state_e return_val;
	log_vine_task_wait( task,__FUNCTION__,task_duration,return_val);
}
