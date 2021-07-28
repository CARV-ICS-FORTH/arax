#include "arch/alloc.h"
#include "utils/timer.h"
#include <string.h>
#define MALLOC_INSPECT_ALL 1
#include <malloc.h>
#define ONLY_MSPACES       1
#define USE_SPIN_LOCKS     1
#define MSPACES            1
#define HAVE_MMAP          1
#include "malloc.h"

struct arch_alloc_inner_s
{
    arch_alloc_s   base;
    mspace *       root;
    void *         start;
    utils_bitmap_s bmp;
};

#define BIT_ALLOCATOR_BLOCK      (4096ul)
#define BIT_ALLOCATOR_BLOCK_MASK (BIT_ALLOCATOR_BLOCK - 1)
#define BITS_PER_PAGE            (BIT_ALLOCATOR_BLOCK * 8ul)


static struct arch_alloc_inner_s *global_alloc;

int arch_alloc_init_once(arch_alloc_s *_alloc, size_t size)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;

    void *usable_area = alloc + 1;

    void *alligned_start =
      (void *) ((((size_t) usable_area) + (BIT_ALLOCATOR_BLOCK_MASK)) & (~BIT_ALLOCATOR_BLOCK_MASK));

    size_t alligned_size = size - (((size_t) alligned_start) - ((size_t) _alloc));

    size_t alligned_pages = alligned_size / BIT_ALLOCATOR_BLOCK;

    size_t bitmap_pages = (alligned_pages + (BITS_PER_PAGE - 1)) / BITS_PER_PAGE;

    alligned_start += BIT_ALLOCATOR_BLOCK * bitmap_pages;

    alligned_pages -= bitmap_pages;

    utils_bitmap_init(&(alloc->bmp), alligned_pages);

    alloc->start = alligned_start;

    global_alloc = alloc;

    alloc->root = create_mspace(0, 1);

    return 0;
}

void arch_alloc_init_always(arch_alloc_s *alloc)
{
    global_alloc = (struct arch_alloc_inner_s *) alloc;
}

void* arch_alloc_allocate(arch_alloc_s *_alloc, size_t size)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;
    void *data;

    #ifdef ALLOC_STATS
    utils_timer_s dt;
    utils_timer_set(dt, start);
    #endif

    data = mspace_malloc(alloc->root, size);

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

    mspace_free(alloc->root, mem);

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

    destroy_mspace(alloc->root);
}

static void _arch_alloc_mspace_mallinfo(mspace *mspace, arch_alloc_stats_s *stats)
{
    struct mallinfo minfo = mspace_mallinfo(mspace);

    stats->total_bytes += (unsigned int) minfo.arena;
    stats->used_bytes  += (unsigned int) minfo.uordblks;
}

arch_alloc_stats_s arch_alloc_stats(arch_alloc_s *_alloc)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;
    arch_alloc_stats_s stats         = { 0 };

    _arch_alloc_mspace_mallinfo(alloc->root, &stats);

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

void arch_alloc_inspect(arch_alloc_s *_alloc, void (*inspector)(void *start, void *end, size_t size,
  void *arg), void *arg)
{
    struct arch_alloc_inner_s *alloc = (struct arch_alloc_inner_s *) _alloc;

    mspace_inspect_all(alloc->root, inspector, arg);
}

void* vine_mmap(size_t s)
{
    vine_assert(!(s & BIT_ALLOCATOR_BLOCK_MASK));
    s /= BIT_ALLOCATOR_BLOCK;
    size_t off = utils_bitmap_alloc_bits(&(global_alloc->bmp), s);

    vine_assert(off != BITMAP_NOT_FOUND);
    off *= BIT_ALLOCATOR_BLOCK;
    return global_alloc->start + off;
}

void* vine_ummap(void *a, size_t s)
{
    size_t start       = (a - (global_alloc->start)) / BIT_ALLOCATOR_BLOCK;
    size_t size_blocks = s / BIT_ALLOCATOR_BLOCK;

    utils_bitmap_free_bits(&(global_alloc->bmp), start, size_blocks);
    return 0;
}

utils_bitmap_s* arch_alloc_get_bitmap()
{
    return &(global_alloc->bmp);
}
