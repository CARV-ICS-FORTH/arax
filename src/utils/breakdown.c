#include "breakdown.h"
#ifdef BREAKS_ENABLE
#include <stdio.h>

#ifdef VINE_TELEMETRY

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "config.h"
#include <stdlib.h>

int collector_fd;

void utils_breakdown_init_telemetry(char * conf)
{
	char telemetry_host[256];
	int telemetry_port;
	static struct sockaddr_in serv_addr ={0};

	collector_fd = socket(PF_INET,SOCK_STREAM,0);
	serv_addr.sin_family = PF_INET;
	utils_config_get_int(conf,"telemetry_port",&telemetry_port,8889);
	serv_addr.sin_port = htons(telemetry_port);
	utils_config_get_str(conf,"telemetry_host",telemetry_host,256,"127.0.0.1");
	serv_addr.sin_addr.s_addr = inet_addr(telemetry_host);

	if(connect(collector_fd,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) < 0)
	{
		printf("Connection to collector %s:%d could not be established.\n",telemetry_host,telemetry_port);
		abort();
	}
	else
		printf("Connection to collector %s:%d established.\n",telemetry_host,telemetry_port);
}
#endif


void utils_breakdown_init_stats(utils_breakdown_stats_s * stats)
{
	memset(stats,0,sizeof(*stats));
	stats->head_ptr = stats->heads;
}

void utils_breakdown_instance_init(utils_breakdown_instance_s * bdown)
{
	memset(bdown,0,sizeof(*bdown));
}

unsigned long long int get_now_ns()
{
	struct timespec now;

	clock_gettime(CLOCK_REALTIME,&now);

	return (now.tv_sec*1000000000+now.tv_nsec);
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
		#ifdef VINE_TELEMETRY
			bdown->start = get_now_ns();
		#endif
	}
	else
	{
		unsigned long long now = get_now_ns();
		unsigned long long last;

		#ifdef VINE_TELEMETRY
			bdown->start = now;
		#endif

		last = __sync_lock_test_and_set(&(stats->last),now);
		if(last)
			__sync_fetch_and_add(stats->part+BREAKDOWN_PARTS+1,now-last);
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
//		bdown->stats->head_ptr += sprintf(bdown->stats->head_ptr," %s,",description);
		if(bdown->stats->head_ptr)
			__sync_fetch_and_add(bdown->stats->head_ptr,sprintf(bdown->stats->head_ptr," %s,",description));
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
	if(bdown->first)
		bdown->stats->head_ptr = 0;

	unsigned long long now = get_now_ns();
	__sync_lock_test_and_set(&(bdown->stats->last),now);

#ifdef VINE_TELEMETRY
	send(collector_fd,bdown,sizeof(*bdown),0);
#endif
}

unsigned long long utils_breakdown_duration(utils_breakdown_instance_s * bdown)
{
	return bdown->part[BREAKDOWN_PARTS];
}

#endif
