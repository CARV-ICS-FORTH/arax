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

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Initialize bitmap starting in \c mem, holding \c size_bits bits.
 *
 * \note Ensure \c mem is greater than or equal to \c UTILS_BITMAP_CALC_BYTES()
 *
 * \param mem Pointer to memory of \c UTILS_BITMAP_CALC_BYTES(size_bits) or more
 * \param size_bits Number of bits this bitmap will hold.
 * \return Returns mem on success, null otherwise.
 */
utils_bitmap_s* utils_bitmap_init(void *mem, size_t size_bits);

/**
 * Returns number of bits \c bmp holds, same as the value given in \c utils_bitmap_init.
 *
 * \c bmp An initialized utils_bitmap_s instance.
 * \return Bits contained in \c bmp
 */
size_t utils_bitmap_size(utils_bitmap_s *bmp);

/**
 * Returns number of bits currently used in \c bmp.
 *
 * \c bmp An initialized utils_bitmap_s instance.
 * \return Number of allocated bits in \c bmp.
 */
size_t utils_bitmap_used(utils_bitmap_s *bmp);

/**
 * Returns number of unused bits in \c bmp.
 *
 * \note This is utils_bitmap_size() - utils_bitmap_used()
 *
 * \c bmp An initialized utils_bitmap_s instance.
 * \return Number of free bits in \c bmp.
 */
size_t utils_bitmap_free(utils_bitmap_s *bmp);

/**
 * Allocate \c bits cotiguous bits from \c bmp and return index of first bit.
 *
 * \c bmp An initialized utils_bitmap_s instance.
 * \return Bit index of allocation start.
 */
size_t utils_bitmap_alloc_bits(utils_bitmap_s *bmp, size_t bits);

/**
 * Free contiguous \c bits starting from \c start, from the \c bmp bitmap.
 *
 * \c bmp An initialized utils_bitmap_s instance.
 * \c start First bit index to be freed.
 * \c bits Number of bits to free.
 */
void utils_bitmap_free_bits(utils_bitmap_s *bmp, size_t start, size_t bits);

/**
 * Count allocated bits of \c bmp.
 *
 * Should always return the same value as \c utils_bitmap_used().
 *
 * \note This should only be used in tests, as it is very slow.
 *
 * \c bmp An initialized utils_bitmap_s instance.
 * \return Bits allocated(set) in \c bits array of \c bmp
 */
size_t utils_bitmap_count_allocated(utils_bitmap_s *bmp);

/**
 * Print the bitmap oc \c bmp, in stderr.
 *
 * \note Use this for debuging
 *
 * \c bmp An initialized utils_bitmap_s instance.
 */
void utils_bitmap_print_bits(utils_bitmap_s *bmp);
#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif // ifndef UTILS_BITMAP_HEADER
