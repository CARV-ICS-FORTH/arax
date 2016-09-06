#ifdef BREAKS_ENABLE
#include "breakdown.h"
#include <stdio.h>

void utils_breakdown_init_stats(utils_breakdown_stats_s * stats)
{
	memset(stats,0,sizeof(*stats));
}

void utils_breakdown_begin(utils_breakdown_instance_s * bdown,utils_breakdown_stats_s * stats)
{
	bdown->stats = stats;
	__sync_fetch_and_add(&(stats->samples),1);
	bdown->current_part = 0;
	utils_timer_set(bdown->timer,start);	// Start counting
}

void utils_breakdown_advance(utils_breakdown_instance_s * bdown)
{
	int current;
	utils_timer_set(bdown->timer,stop);
	current = __sync_fetch_and_add(&(bdown->current_part),1);
	__sync_fetch_and_add(bdown->stats->part+current,utils_timer_get_duration_ns(bdown->timer));
	// Pick up right where we left of
	utils_timer_set_raw(bdown->timer,start,utils_timer_get_raw(bdown->timer,stop));
}

void utils_breakdown_end(utils_breakdown_instance_s * bdown)
{
	int current;
	utils_timer_set(bdown->timer,stop);
	current = __sync_fetch_and_add(&(bdown->current_part),1);
	__sync_fetch_and_add(bdown->stats->part+current,utils_timer_get_duration_ns(bdown->timer));
}

void utils_breakdown_write(const char *file,vine_accel_type_e type,const char * description,utils_breakdown_stats_s * stats)
{
	FILE * f = fopen(file,"a");
	int part;

	fprintf(f,"%d,%s,%llu",type,description,stats->samples);
	for(part = 0 ; part < BREAKDOWN_PARTS ; ++part)
	{
		fprintf(f,",%llu",stats->part[part]);
	}
	fputs("\n",f);
	fclose(f);
}

#endif
