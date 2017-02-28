#include "breakdown.h"
#ifdef BREAKS_ENABLE
#include <stdio.h>

void utils_breakdown_init_stats(utils_breakdown_stats_s * stats)
{
	memset(stats,0,sizeof(*stats));
	stats->head_ptr = stats->heads;
}

void utils_breakdown_begin(utils_breakdown_instance_s * bdown,utils_breakdown_stats_s * stats,const char * description)
{
	bdown->stats = stats;
	__sync_fetch_and_add(&(stats->samples),1);

	if(stats->head_ptr)
	{
		bdown->stats->desc[0] = bdown->stats->head_ptr;
		bdown->stats->head_ptr += sprintf(bdown->stats->head_ptr," %s,",description);
	}
	bdown->current_part = 0;
	bdown->part[BREAKDOWN_PARTS] = 0;
	utils_timer_set(bdown->timer,start);	// Start counting
}

void utils_breakdown_advance(utils_breakdown_instance_s * bdown,const char * description)
{
	int current;
	utils_timer_set(bdown->timer,stop);
	current = __sync_fetch_and_add(&(bdown->current_part),1);
	bdown->part[current] = utils_timer_get_duration_ns(bdown->timer);
	bdown->part[BREAKDOWN_PARTS] += bdown->part[current];
	if(bdown->stats->head_ptr)
	{
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
	bdown->part[current] = utils_timer_get_duration_ns(bdown->timer);
	bdown->part[BREAKDOWN_PARTS] += bdown->part[current];
	for(cnt = 0 ; cnt < current ; cnt++)	// Update per proc breakdown
		__sync_add_and_fetch(bdown->stats->part+cnt,bdown->part[current]);
	bdown->stats->head_ptr = 0;
}

void utils_breakdown_write(const char *file,vine_accel_type_e type,const char * description,utils_breakdown_stats_s * stats)
{
	FILE * f;
	char ffile[1024];
	int part,uparts;

	if(!stats->samples)
		return; /* Do not write anything  */

	snprintf(ffile,1024,"%s.hdr",file);
	f = fopen(ffile,"a");
	// Print header and count parts
	fprintf(f,"TYPE, PROC, SAMPLES,%s\n",stats->heads);
	fclose(f);

	snprintf(ffile,1024,"%s.brk",file);
	f = fopen(ffile,"a");
	fprintf(f,"%s,%s,%llu",vine_accel_type_to_str(type),description,stats->samples);
	for(uparts = BREAKDOWN_PARTS-1 ; uparts >= 0 ; uparts++)
		if(!stats->part[uparts])
			break;
	for(part = 0 ; part < uparts ; ++part)
		fprintf(f,",%llu",stats->part[part]);
	fputs("\n",f);
	fclose(f);
}

unsigned long long utils_breakdown_duration(utils_breakdown_instance_s * bdown)
{
	return bdown->part[BREAKDOWN_PARTS];
}

#endif
