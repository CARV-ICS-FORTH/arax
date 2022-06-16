#include "arch/alloc.h"
#include "tlsf.h"

struct arch_alloc_inner_s
{
    arch_alloc_s   base;
    tlsf_t         root;
    utils_spinlock lock;
};

int arch_alloc_init_once(arch_alloc_s *_alloc, size_t size)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;

    void *usable_area = alloc + 1;

    utils_spinlock_init(&(alloc->lock));
    alloc->root = tlsf_create_with_pool(usable_area, size - sizeof(struct arch_alloc_inner_s));

    return 0;
}

void arch_alloc_init_always(arch_alloc_s *alloc)
{ }

void* arch_alloc_allocate(arch_alloc_s *_alloc, size_t size)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;
    void *data;

    #ifdef ALLOC_STATS
    utils_timer_s dt;
    utils_timer_set(dt, start);
    #endif

    utils_spinlock_lock(&(alloc->lock));
    data = tlsf_malloc(alloc->root, size);
    utils_spinlock_unlock(&(alloc->lock));

    #ifdef ALLOC_STATS
    utils_timer_set(dt, stop);
    __sync_fetch_and_add(&(_alloc->alloc_ns[!!data]),
      utils_timer_get_duration_ns(dt));
    __sync_fetch_and_add(&(_alloc->allocs[!!data]), 1);
    #endif
    return data;
}

void _arch_alloc_free(arch_alloc_s *_alloc, void *mem)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;

    #ifdef ALLOC_STATS
    utils_timer_s dt;
    utils_timer_set(dt, start);
    #endif

    utils_spinlock_lock(&(alloc->lock));
    tlsf_free(alloc->root, mem);
    utils_spinlock_unlock(&(alloc->lock));

    #ifdef ALLOC_STATS
    utils_timer_set(dt, stop);
    __sync_fetch_and_add(&(_alloc->free_ns),
      utils_timer_get_duration_ns(dt));
    __sync_fetch_and_add(&(_alloc->frees), 1);
    #endif
}

void arch_alloc_exit(arch_alloc_s *_alloc)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;

    tlsf_destroy(alloc->root);
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s *_alloc)
{
    arch_alloc_stats_s stats = { 0 };

    #ifdef ALLOC_STATS
    stats.allocs[0]   = _alloc->allocs[0];
    stats.allocs[1]   = _alloc->allocs[1];
    stats.frees       = _alloc->frees;
    stats.alloc_ns[0] = _alloc->alloc_ns[0];
    stats.alloc_ns[1] = _alloc->alloc_ns[1];
    stats.free_ns     = _alloc->free_ns;
    #endif
    return stats;
}

struct inspect_walker_state
{
    void  (*inspector)(void *start, void *end, size_t size, void *arg);
    void *arg;
};

void inspect_walker(void *ptr, size_t size, int used, void *user)
{
    struct inspect_walker_state *iws = (struct inspect_walker_state *) user;

    iws->inspector(ptr, ptr + size, used * size, iws->arg);
}

void arch_alloc_inspect(arch_alloc_s *_alloc, void (*inspector)(void *start, void *end, size_t size,
  void *arg), void *arg)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;
    struct inspect_walker_state iws  = { inspector, arg };

    tlsf_walk_pool(tlsf_get_pool(alloc->root), inspect_walker, &iws);
}

void* vine_mmap(size_t s)
{
    return 0;
}

void* vine_ummap(void *a, size_t s)
{
    return 0;
}

arch_alloc_s* arch_alloc_create_sub_alloc(arch_alloc_s *parent)
{
    return parent;
}

utils_bitmap_s* arch_alloc_get_bitmap()
{
    return 0;
}
