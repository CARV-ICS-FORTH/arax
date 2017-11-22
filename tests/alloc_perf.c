#include "arch/alloc.h"
#include "utils/timer.h"
#include "testing.h"
#include <sys/time.h>

#define SCALE_CORES 16

#define POOL_SIZE 0x40000000
#define ALLOC_COUNT  80000
#define ALLOC_SIZE  10000
arch_alloc_s * alloc;
char * ma = 0;
void setup()
{
	int cnt = 0;
	test_backup_config();
	unlink("/dev/shm/vt_test"); /* Start fresh */

	ma = malloc(POOL_SIZE+16);
	*(uint64_t*)ma = 0x0DDF00DBADC0FFEE;
	ma += 8;
	alloc = (arch_alloc_s *)ma;
	printf("Pool size: %d\n",POOL_SIZE);
	for(cnt = 0 ; cnt < POOL_SIZE ; cnt += 1024)
	{
		ma[cnt] = 0;
	}
	*(uint64_t*)(ma+POOL_SIZE) = 0xBADC0FFEE0DDF00D;
	arch_alloc_init(alloc,POOL_SIZE);
	printf("Total operations: %d\n",ALLOC_COUNT);
	printf("Allocation Size: %d\n",ALLOC_SIZE);
	printf("%16s,%16s,%16s,%16s,%16s\n","Threads","Alloc Cpu Time","Free Cpu Time","Alloc Clock Time","Free Clock Time");
}

void teardown()
{
	arch_alloc_exit(alloc);
	ck_assert_int_eq(*(uint64_t*)(ma+POOL_SIZE),0xBADC0FFEE0DDF00D);
	ma -= 8;
	ck_assert_int_eq(*(uint64_t*)(ma),0x0DDF00DBADC0FFEE);
	free(ma);
}


void print_alloc_info()
{
	int cnt;
	arch_alloc_stats_s stats = arch_alloc_stats(alloc);
	size_t * sr = (size_t*)&stats;
	const char * strs[9] =
	{
		"total_bytes  :",
		"used_bytes   :",
		"mspaces      :",
		"alloc fail   :",
		"alloc good   :",
		"free         :",
		"alloc fail us:",
		"alloc good us:",
		"free       us:"
	};
	for(cnt = 0 ; cnt < sizeof(arch_alloc_stats_s)/sizeof(size_t) ; cnt++)
	{
		printf("%s %lu\n",strs[cnt],*sr);
		sr++;
	}
}

volatile int synchro = 1;

struct timers
{
	size_t alloc_d;
	size_t free_d;
};

void * alloc_thread(void * data)
{
	size_t allocs = (size_t)data;
	void ** mems = malloc(sizeof(void*)*allocs);
	struct timers * t = malloc(sizeof(struct timers));
	memset(t,0,sizeof(struct timers));
	int cnt;
	utils_timer_s timer;

	while(synchro);

	utils_timer_set(timer,start);
	for(cnt = 0 ; cnt < allocs ; cnt++)
	{
		mems[cnt] = arch_alloc_allocate(alloc,ALLOC_SIZE);
		ck_assert(!!mems[cnt]);
	}
	utils_timer_set(timer,stop);
	t->alloc_d = utils_timer_get_duration_ns(timer);

	__sync_fetch_and_add(&synchro,1);
	while(synchro);

	utils_timer_set(timer,start);
	for(cnt = 0 ; cnt < allocs ; cnt++)
	{
		arch_alloc_free(alloc,mems[cnt]);
	}
	utils_timer_set(timer,stop);
	t->free_d = utils_timer_get_duration_ns(timer);

	free(mems);
	return t;
}

START_TEST(alloc_perf)
{
	int cnt;
	pthread_t * threads = malloc(_i*sizeof(pthread_t));
	struct timers ** ts = malloc(_i*sizeof(struct timers *));
	struct timers batch;
	utils_timer_s timer;

	for(cnt = 0 ; cnt < _i ; cnt++)
		pthread_create(threads+cnt,0,alloc_thread,(void*)(size_t)(ALLOC_COUNT/_i));

	usleep(100);
	synchro = 0;
	utils_timer_set(timer,start);
	while(synchro != _i)
		usleep(100);
	utils_timer_set(timer,stop);
	batch.alloc_d = utils_timer_get_duration_ns(timer);
	usleep(1000);
	synchro = 0;
	utils_timer_set(timer,start);
	for(cnt = 0 ; cnt < _i ; cnt++)
		pthread_join(threads[cnt],(void**)(ts+cnt));
	utils_timer_set(timer,stop);
	batch.free_d = utils_timer_get_duration_ns(timer);


	for(cnt = 1 ; cnt < _i ; cnt++)
	{
		ts[0]->alloc_d += ts[cnt]->alloc_d;
		ts[0]->free_d += ts[cnt]->free_d;
		free(ts[cnt]);
	}
	printf("%16d,%16lu,%16lu,%16lu,%16lu\n",_i,ts[0]->alloc_d/_i,ts[0]->free_d/_i,batch.alloc_d,batch.free_d);
	print_alloc_info();
	free(ts);
	free(threads);

}
END_TEST

Suite* suite_init()
{
	Suite *s;
	TCase *tc_multi;

	s         = suite_create("Perf");
	tc_multi = tcase_create("Multi");
	tcase_add_unchecked_fixture(tc_multi, setup, teardown);
	tcase_add_loop_test(tc_multi, alloc_perf, 1, SCALE_CORES+1);
	tcase_set_timeout(tc_multi,30);
	suite_add_tcase(s, tc_multi);
	return s;
}

int main(int argc, char *argv[])
{
	Suite   *s;
	SRunner *sr;
	int     failed;

	s  = suite_init();
	sr = srunner_create(s);
	srunner_set_fork_status(sr, CK_NOFORK);
	srunner_run_all(sr, CK_NORMAL);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
