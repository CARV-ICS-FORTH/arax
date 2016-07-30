#include "arch/alloc.h"
#include "testing.h"
#include <pthread.h>
#include <sys/time.h>
#include <pthread.h>

#define SCALE_CORES 16

#define POOL_SIZE 0x20000000
#define ALLOC_COUNT  500000
#define ALLOC_SIZE  1000
arch_alloc_s * alloc = 0;
char * ma = 0;
void setup()
{
	int cnt = 0;
	test_backup_config();
	unlink("/dev/shm/vt_test"); /* Start fresh */

	ma = malloc(POOL_SIZE);
	printf("Pool size: %d\n",POOL_SIZE);
	for(cnt = 0 ; cnt < POOL_SIZE ; cnt += 1024)
	{
		ma[cnt] = 0;
	}
	alloc = arch_alloc_init(ma,POOL_SIZE);
	printf("Total operations: %d\n",ALLOC_COUNT);
	printf("Allocation Size: %d\n",ALLOC_SIZE);
	printf("%16s,%16s,%16s,%16s,%16s\n","Threads","Alloc Cpu Time","Free Cpu Time","Alloc Clock Time","Free Clock Time");
}

void teardown()
{
	arch_alloc_exit(alloc);
	free(ma);
}

volatile int synchro = 1;

struct times
{
	int alloc_d;
	int free_d;
};

void * alloc_thread(void * data)
{
	size_t allocs = (size_t)data;
	void ** mems = malloc(sizeof(void*)*allocs);
	struct times * t = malloc(sizeof(struct times));
	memset(t,0,sizeof(struct times));
	int cnt;
	struct timeval start,end;

	while(synchro);

	gettimeofday(&start,0);
	for(cnt = 0 ; cnt < allocs ; cnt++)
	{
		mems[cnt] = arch_alloc_allocate(alloc,ALLOC_SIZE);
		ck_assert(mems[cnt]);
	}
	gettimeofday(&end,0);
	t->alloc_d =  (end.tv_sec - start.tv_sec) * 1000000;
	t->alloc_d += (end.tv_usec - start.tv_usec);

	__sync_fetch_and_add(&synchro,1);
	while(synchro);

	gettimeofday(&start,0);
	for(cnt = 0 ; cnt < allocs ; cnt++)
	{
		arch_alloc_free(alloc,mems[cnt]);
	}
	gettimeofday(&end,0);
	t->free_d =  (end.tv_sec - start.tv_sec) * 1000000;
	t->free_d += (end.tv_usec - start.tv_usec);

	free(mems);
	return t;
}

START_TEST(alloc_perf)
{
	int cnt;
	pthread_t * threads = malloc(_i*sizeof(pthread_t));
	struct times ** ts = malloc(_i*sizeof(struct times *));
	struct timeval start,end;
	int alloc_d;
	int free_d;
	for(cnt = 0 ; cnt < _i ; cnt++)
		pthread_create(threads+cnt,0,alloc_thread,(void*)(size_t)(ALLOC_COUNT/_i));

	usleep(1000);
	synchro = 0;
	gettimeofday(&start,0);
	while(synchro != _i)
		usleep(10000);
	gettimeofday(&end,0);
	alloc_d =  (end.tv_sec - start.tv_sec) * 1000000;
	alloc_d += (end.tv_usec - start.tv_usec);

	usleep(1000);
	synchro = 0;
	gettimeofday(&start,0);
	for(cnt = 0 ; cnt < _i ; cnt++)
		pthread_join(threads[cnt],(void**)(ts+cnt));
	gettimeofday(&end,0);
	free_d =  (end.tv_sec - start.tv_sec) * 1000000;
	free_d += (end.tv_usec - start.tv_usec);


	for(cnt = 1 ; cnt < _i ; cnt++)
	{
		ts[0]->alloc_d += ts[cnt]->alloc_d;
		ts[0]->free_d += ts[cnt]->free_d;
		free(ts[cnt]);
	}
	printf("%16d,%16d,%16d,%16d,%16d\n",_i,ts[0]->alloc_d/_i,ts[0]->free_d/_i,alloc_d,free_d);
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
	srunner_run_all(sr, CK_NORMAL);
	failed = srunner_ntests_failed(sr);
	srunner_free(sr);
	return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
