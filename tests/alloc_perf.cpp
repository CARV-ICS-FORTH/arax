#include "arch/alloc.h"
#include "utils/timer.h"
#include "testing.h"
#include <sys/time.h>

#define SCALE_CORES 16

#define POOL_SIZE   0x40000000
#define ALLOC_COUNT 80000
#define ALLOC_SIZE  10000

arch_alloc_s *alloc;

void print_alloc_info(arch_alloc_s *alloc)
{
    int cnt;
    arch_alloc_stats_s stats = arch_alloc_stats(alloc);
    size_t *sr = (size_t *) &stats;
    const char *strs[9] = {
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

    for (cnt = 0; cnt < sizeof(arch_alloc_stats_s) / sizeof(size_t); cnt++) {
        printf("%s %lu\n", strs[cnt], *sr);
        sr++;
    }
}

volatile int start_sync = 1;
volatile int synchro    = 0;

struct timers
{
    size_t alloc_d;
    size_t free_d;
};

void* alloc_thread(void *data)
{
    size_t allocs    = (size_t) data;
    void **mems      = new void *[allocs];
    struct timers *t = new struct timers ();

    memset(t, 0, sizeof(struct timers));
    int cnt;
    utils_timer_s timer;

    while (start_sync);

    utils_timer_set(timer, start);
    for (cnt = 0; cnt < allocs; cnt++) {
        mems[cnt] = arch_alloc_allocate(alloc, ALLOC_SIZE);
        REQUIRE(mems[cnt]);
    }
    utils_timer_set(timer, stop);
    t->alloc_d = utils_timer_get_duration_ns(timer);

    __sync_fetch_and_add(&synchro, 1);
    while (synchro);

    utils_timer_set(timer, start);
    for (cnt = 0; cnt < allocs; cnt++) {
        arch_alloc_free(alloc, mems[cnt]);
    }
    utils_timer_set(timer, stop);
    t->free_d = utils_timer_get_duration_ns(timer);

    delete [] mems;
    return t;
}

struct inspector_state
{
    void * test_allocation;
    void * start;
    void * end;
    size_t size;
};

void inspector(void *start, void *end, size_t size, void *arg)
{
    struct inspector_state *ist = (struct inspector_state *) arg;

    if (start == ist->test_allocation) {
        ist->start = start;
        ist->end   = end;
        ist->size  = size;
        printf("Allocation %p %p %lu %p\n", start, end, size, arg);
    }
}

TEST_CASE("Alloc Perf")
{
    char *ma = 0;

    int cnt = 0;

    test_common_setup("");
    unlink("/dev/shm/vt_test"); /* Start fresh */

    ma = new char[POOL_SIZE + 16];
    *(uint64_t *) ma = 0x0DDF00DBADC0FFEE;
    ma   += 8;
    alloc = (arch_alloc_s *) ma;
    printf("Pool size: %d\n", POOL_SIZE);
    for (cnt = 0; cnt < POOL_SIZE; cnt += 1024) {
        ma[cnt] = 0;
    }
    *(uint64_t *) (ma + POOL_SIZE) = 0xBADC0FFEE0DDF00D;
    arch_alloc_init_once(alloc, POOL_SIZE);
    printf("Total operations: %d\n", ALLOC_COUNT);
    printf("Allocation Size: %d\n", ALLOC_SIZE);
    printf("%16s,%16s,%16s,%16s,%16s\n", "Threads", "Alloc Cpu Time", "Free Cpu Time", "Alloc Clock Time",
      "Free Clock Time");

    for (int nthreads = 1; nthreads <= SCALE_CORES; nthreads++) {
        DYNAMIC_SECTION("alloc_perf Threads:" << nthreads)
        {
            int cnt;
            pthread_t *threads = new pthread_t[nthreads];
            struct timers **ts = new timers *[nthreads];
            struct timers batch;
            utils_timer_s timer;

            for (cnt = 0; cnt < nthreads; cnt++)
                pthread_create(threads + cnt, 0, alloc_thread, (void *) (size_t) (ALLOC_COUNT / nthreads));

            safe_usleep(100);
            start_sync = 0;
            utils_timer_set(timer, start);
            while (synchro != nthreads)
                safe_usleep(100);
            utils_timer_set(timer, stop);
            batch.alloc_d = utils_timer_get_duration_ns(timer);
            safe_usleep(1000);
            synchro = 0;
            utils_timer_set(timer, start);
            for (cnt = 0; cnt < nthreads; cnt++)
                pthread_join(threads[cnt], (void **) (ts + cnt));
            utils_timer_set(timer, stop);
            batch.free_d = utils_timer_get_duration_ns(timer);


            for (cnt = 1; cnt < nthreads; cnt++) {
                ts[0]->alloc_d += ts[cnt]->alloc_d;
                ts[0]->free_d  += ts[cnt]->free_d;
                delete ts[cnt];
            }

            printf("%16d,%16lu,%16lu,%16lu,%16lu\n", nthreads, ts[0]->alloc_d / nthreads, ts[0]->free_d / nthreads,
              batch.alloc_d, batch.free_d);
            print_alloc_info(alloc);

            delete ts[0];
            delete [] ts;
            delete [] threads;
        }
    }

    SECTION("test_inspect")
    {
        void *test_allocation = arch_alloc_allocate(alloc, ALLOC_SIZE);

        struct inspector_state ist = { test_allocation, 0 };

        printf("Allocation %p\n", test_allocation);

        arch_alloc_inspect(alloc, inspector, (void *) &ist);

        REQUIRE(ist.start == test_allocation);
        REQUIRE(ist.size >= ALLOC_SIZE);

        arch_alloc_free(alloc, test_allocation);
    }

    SECTION("test_sub_allocator")
    {
        arch_alloc_s *sub = arch_alloc_create_sub_alloc(alloc);

        REQUIRE(sub);

        void *ptr = arch_alloc_allocate(sub, 1000);

        REQUIRE(ptr);

        arch_alloc_free(sub, ptr);

        arch_alloc_exit(sub);
    }

    test_common_teardown();
    arch_alloc_exit(alloc);
    REQUIRE(*(uint64_t *) (ma + POOL_SIZE) == 0xBADC0FFEE0DDF00D);
    ma -= 8;
    REQUIRE(*(uint64_t *) (ma) == 0x0DDF00DBADC0FFEE);
    delete [] ma;
}
