#include "breakdown.h"
#ifdef BREAKS_ENABLE
#include <stdio.h>

void utils_breakdown_init_stats(utils_breakdown_stats_s * stats)
{
	memset(stats,0,sizeof(*stats));
	stats->head_ptr = stats->heads;
}

void utils_breakdown_instance_init(utils_breakdown_instance_s * bdown)
{
	memset(bdown,0,sizeof(*bdown));
}

void utils_breakdown_begin(utils_breakdown_instance_s * bdown,utils_breakdown_stats_s * stats,const char * description)
{
	bdown->stats = stats;
	bdown->first = !__sync_fetch_and_add(&(stats->samples),1);
	bdown->part[BREAKDOWN_PARTS] = 0;
	if(bdown->first)
	{
		stats->desc[0] = stats->head_ptr;
		stats->head_ptr += sprintf(stats->head_ptr," %s,",description);
	}
	else
	{
		utils_timer_set(stats->interval,stop);
		__sync_fetch_and_add(stats->part+BREAKDOWN_PARTS+1,utils_timer_get_duration_ns(stats->interval));
	}
	bdown->current_part = 0;
	utils_timer_set(bdown->timer,start);	// Start counting
}

void utils_breakdown_advance(utils_breakdown_instance_s * bdown,const char * description)
{
	int current;
	utils_timer_set(bdown->timer,stop);
	current = __sync_fetch_and_add(&(bdown->current_part),1);
	__sync_fetch_and_add(bdown->part+current,utils_timer_get_duration_ns(bdown->timer));
	__sync_fetch_and_add(bdown->part+BREAKDOWN_PARTS,bdown->part[current]);

	if(bdown->first)
	{	// There can be only one (first)
		bdown->stats->desc[current+1] = bdown->stats->head_ptr;
		bdown->stats->head_ptr += sprintf(bdown->stats->head_ptr," %s,",description);
	}

	// Pick up right where we left of
	utils_timer_set_raw(bdown->timer,start,utils_timer_get_raw(bdown->timer,stop));
}

void utils_breakdown_end(utils_breakdown_instance_s * bdown)
{
	int current;
	int cnt;
	utils_timer_set(bdown->timer,stop);
	current = __sync_fetch_and_add(&(bdown->current_part),1);
	__sync_fetch_and_add(bdown->part+current,utils_timer_get_duration_ns(bdown->timer));
	__sync_fetch_and_add(bdown->part+BREAKDOWN_PARTS,bdown->part[current]);
	for(cnt = 0 ; cnt <= current ; cnt++)	// Update per proc breakdown
		__sync_add_and_fetch(bdown->stats->part+cnt,bdown->part[cnt]);
	__sync_add_and_fetch(bdown->stats->part+BREAKDOWN_PARTS,bdown->part[BREAKDOWN_PARTS]);
	bdown->stats->head_ptr = 0;
	utils_timer_set(bdown->stats->interval,start);
}

unsigned long long utils_breakdown_duration(utils_breakdown_instance_s * bdown)
{
	return bdown->part[BREAKDOWN_PARTS];
}

#endif
