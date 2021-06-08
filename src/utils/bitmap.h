#ifndef UTILS_BITMAP_HEADER
#define UTILS_BITMAP_HEADER
#include "spinlock.h"
#include <stddef.h>

typedef struct utils_bitmap
{
    utils_spinlock lock;
    uint64_t *     free_p;
    uint64_t *     end;
    size_t         used_bits;
    size_t         size_bits;
    uint64_t       bits[];
} utils_bitmap_s;

#define UTILS_BITMAP_CALC_BYTES(BITS) ((sizeof(utils_bitmap_s) + sizeof(uint64_t) * ((BITS + 63) / 64)))

#define BITMAP_NOT_FOUND ((size_t) -1)

utils_bitmap_s* utils_bitmap_init(void *mem, size_t size_bits);

size_t utils_bitmap_size(utils_bitmap_s *bmp);

size_t utils_bitmap_used(utils_bitmap_s *bmp);

size_t utils_bitmap_free(utils_bitmap_s *bmp);

size_t utils_bitmap_alloc_bits(utils_bitmap_s *bmp, size_t bits);

void utils_bitmap_free_bits(utils_bitmap_s *bmp, size_t start, size_t bits);

/**
 * Count allocated bits of \c bmp.
 *
 * Should always return the same value as \c utils_bitmap_used().
 *
 * \note This should only be used in tests, as it is very slow.
 *
 * \param bmp A valid utils_bitmap_s instance.
 * \return Bits allocated(set) in \c bits array of \c bmp
 */
size_t utils_bitmap_count_allocated(utils_bitmap_s *bmp);

/**
 * Print the bitmap oc \c bmp, in stderr.
 *
 * \note Use this for debuging
 */
void utils_bitmap_print_bits(utils_bitmap_s *bmp);
#endif // ifndef UTILS_BITMAP_HEADER
