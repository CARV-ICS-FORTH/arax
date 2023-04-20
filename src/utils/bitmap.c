#include "bitmap.h"
#include <string.h>
#include <stdio.h>

/**
 * A bit about the bitmap:
 *
 * It is arranged in 'chunks' of uint64_t (8 bytes).
 *
 * Initially all bits are 0, meaning free.
 *
 * Allocation happens in squential order for chunks.
 * First chunk[0] will be usedm then chunk[1], etc...
 *
 * Inside a chunk bit allocation occurs in a lsb to msb order:
 *
 * An initialy empty chunk:
 * 0x0000 0000 0000 0000
 * After a 16bit allocation:
 * 0x0000 0000 0000 000F
 * After an 8bit allocation:
 * 0x0000 0000 0000 008F
 */

utils_bitmap_s* utils_bitmap_init(void *mem, size_t size_bits)
{
    utils_bitmap_s *bmp = mem;

    bmp->size_bits = size_bits;

    size_bits += 63;
    size_bits /= 64; // to uint64_t

    bmp->free_p    = bmp->bits;
    bmp->end       = bmp->free_p + size_bits;
    bmp->used_bits = 0;
    utils_spinlock_init(&(bmp->lock));
    memset(bmp->bits, 0, size_bits * 8);

    return bmp;
}

size_t utils_bitmap_size(utils_bitmap_s *bmp)
{
    return bmp->size_bits;
}

size_t utils_bitmap_used(utils_bitmap_s *bmp)
{
    return bmp->used_bits;
}

size_t utils_bitmap_free(utils_bitmap_s *bmp)
{
    return bmp->size_bits - bmp->used_bits;
}

/**
 * Return true if \c bits were free at the begining of the \c chunk
 */
static inline int find_end_small(size_t bits, uint64_t *chunk)
{
    uint64_t mask = BITMAP_NOT_FOUND;

    if (bits != 64)
        mask = (((uint64_t) 1) << bits) - 1;

    if ( (mask & (*chunk)) == 0) { // Yes free
        *chunk |= mask;
        return 1;
    } else { // Nope, already allocated
        return 0;
    }
}

static inline size_t find_start_small(utils_bitmap_s *bmp, size_t bits, uint64_t *chunk)
{
    uint64_t usable_bits = *chunk;

    if (chunk + 1 == bmp->end) { // Final chunk is partially usable
        int shift = bmp->size_bits % 64;
        if (shift)
            usable_bits |= ((uint64_t) -1) << (shift);
    }

    if (!(usable_bits) ) { // Completely empty
        if (bits == 64)
            *chunk = BITMAP_NOT_FOUND;  // Full allocation
        else
            *chunk = (((uint64_t) 1) << bits) - 1;  // Partial allocation
        bmp->used_bits += bits;
        return ( ((size_t) chunk) - ((size_t) bmp->bits) ) * 8;
    }

    if (*chunk == BITMAP_NOT_FOUND) // Completely full
        return BITMAP_NOT_FOUND;

    int free_bits = 64 - __builtin_popcountl(usable_bits);

    if (free_bits >= bits) {
        // Not sure if we have it contigous
        uint64_t mask = (((uint64_t) 1) << bits) - 1;
        int shift;
        for (shift = 0; shift <= 64 - bits; shift++) {
            if ( (mask & (usable_bits) ) == 0)
                goto FOUND;  // Found a good gap
            mask <<= 1;
        }

        return BITMAP_NOT_FOUND;

FOUND:

        *chunk         |= mask;
        bmp->used_bits += bits;

        return ( ((size_t) chunk) - ((size_t) bmp->bits) ) * 8 + shift;
    } else { // Not enough bits in this chunk
        int free_end = (*chunk) ? __builtin_clzl(*chunk) : 64;
        if (free_end == 0) // No free bits at end of chunk
            return BITMAP_NOT_FOUND;

        if (chunk + 1 == bmp->end) // Was on the last chunk
            return BITMAP_NOT_FOUND;

        int remainder = bits - free_end;

        if (find_end_small(remainder, chunk + 1)) {
            *chunk         |= ((uint64_t) -1) << (64 - free_end);
            bmp->used_bits += bits;
            return ( ((size_t) chunk) - ((size_t) bmp->bits) ) * 8 + (64 - free_end);
        }
    }
    return BITMAP_NOT_FOUND;
} /* find_start_small */

static inline size_t find_start_big(utils_bitmap_s *bmp, size_t bits, uint64_t *chunk)
{
    if (*chunk == BITMAP_NOT_FOUND)
        return BITMAP_NOT_FOUND;  // Already full

    uint64_t first_mask = 0;
    size_t remainder;
    int free_end;

    if (*chunk == 0) {
        first_mask = BITMAP_NOT_FOUND; // Allocate the whole chunk
        remainder  = bits - 64;
        free_end   = 0;
    } else {
        free_end   = __builtin_clzl(*chunk);
        first_mask = (free_end) ? (((uint64_t) -1) << (64 - free_end)) : (0);
        remainder  = bits - free_end;
        free_end   = 64 - free_end;
    }

    if (!first_mask)
        return BITMAP_NOT_FOUND;  // No free space at end of chunk

    if (chunk + 1 == bmp->end)
        return BITMAP_NOT_FOUND;  // No next chunk to allocate the remainder

    if (remainder < 65) { // Remainder fits in one chunk
        if (!find_end_small(remainder, chunk + 1))
            return BITMAP_NOT_FOUND;  // No free space at start of next chunk

        *chunk         |= first_mask;
        bmp->used_bits += bits;
        return ( ((size_t) chunk) - ((size_t) bmp->bits) ) * 8 + free_end;
    } else {
        size_t span = remainder / 64;
        size_t last_remainder = remainder & 63;
        uint64_t last_mask    = (((uint64_t) (!!last_remainder)) << (last_remainder)) - 1;
        (void) span;
        (void) last_remainder;

        if (chunk + span + (!!last_remainder) >= bmp->end)
            return BITMAP_NOT_FOUND;  // Not enough chunks

        uint64_t *citr = chunk + 1;
        int c = 0;
        for (; c < span; c++) {
            if (*citr)
                return BITMAP_NOT_FOUND;

            citr++;
        }

        if (last_remainder) {
            if ( (*citr) & last_mask)
                return BITMAP_NOT_FOUND;

            *citr |= last_mask;
        }

        memset(chunk + 1, 0xFF, sizeof(uint64_t) * span);

        *chunk |= first_mask;

        bmp->used_bits += bits;

        return ( ((size_t) chunk) - ((size_t) bmp->bits) ) * 8 + free_end;
    }
    return BITMAP_NOT_FOUND;
} /* find_start_big */

static inline size_t find_start(utils_bitmap_s *bmp, size_t bits, uint64_t *chunk)
{
    if (bits < 65)
        return find_start_small(bmp, bits, chunk);
    else
        return find_start_big(bmp, bits, chunk);

    return BITMAP_NOT_FOUND;
}

size_t utils_bitmap_alloc_bits(utils_bitmap_s *bmp, size_t bits)
{
    uint64_t *chunk;
    size_t found = BITMAP_NOT_FOUND;

    utils_spinlock_lock(&(bmp->lock));

    if (bmp->free_p == bmp->end) {
        bmp->free_p = bmp->bits;
    }

    if (utils_bitmap_free(bmp) < bits)
        goto QUIT;

    for (chunk = bmp->free_p; chunk < bmp->end; chunk++) {
        found = find_start(bmp, bits, chunk);
        if (found != BITMAP_NOT_FOUND)
            goto FOUND;
    }

    for (chunk = bmp->bits; chunk < bmp->free_p; chunk++) {
        found = find_start(bmp, bits, chunk);
        if (found != BITMAP_NOT_FOUND)
            goto FOUND;
    }

    if (found != BITMAP_NOT_FOUND) {
FOUND:

        bmp->free_p = bmp->bits + found / 64;
    } else {
        fprintf(stderr, "%s(%lu) failed (free: %lu)!\n", __func__, bits, utils_bitmap_free(bmp));
    }
QUIT:
    utils_spinlock_unlock(&(bmp->lock));
    return found;
} /* utils_bitmap_alloc_bits */

void utils_bitmap_free_bits(utils_bitmap_s *bmp, size_t start, size_t bits)
{
    utils_spinlock_lock(&(bmp->lock));
    if (bits == 1) {
        bmp->bits[start / 64] &= ~(((uint64_t) 1) << (start & 63));
        bmp->used_bits        -= 1;
    } else {
        size_t end         = start + bits;
        size_t start_chunk = start / 64;
        size_t end_chunk   = (end - 1) / 64;
        start = start & 63;
        end   = end & 63;

        if (start_chunk == end_chunk) { // Single chunk
            size_t mask = BITMAP_NOT_FOUND;
            if (bits != 64)
                mask = ((((uint64_t) 1) << bits) - 1) << (start & 63);

            arax_assert( (bmp->bits[start_chunk] & mask) == mask); // Dont try to free non allocated stuff

            bmp->bits[start_chunk] &= ~mask;

            bmp->used_bits -= bits;
        } else {
            size_t span       = (end_chunk - start_chunk) - 1;
            size_t start_mask = ((uint64_t) -1) << (start);
            size_t end_mask   = (end) ? (((uint64_t) -1) >> ((64 - end))) : (-1);

            arax_assert( (bmp->bits[start_chunk] & start_mask) == start_mask);
            arax_assert( (bmp->bits[end_chunk] & end_mask) == end_mask);

            bmp->bits[start_chunk] &= ~start_mask;
            bmp->bits[end_chunk]   &= ~end_mask;

            if (span)
                memset(bmp->bits + start_chunk + 1, 0, sizeof(uint64_t) * span);

            bmp->used_bits -= bits;
        }
    }
    utils_spinlock_unlock(&(bmp->lock));
} /* utils_bitmap_free_bits */

size_t utils_bitmap_count_allocated(utils_bitmap_s *bmp)
{
    size_t b         = 0;
    size_t allocated = 0;

    for (b = 0; b < (bmp->size_bits + 63) / 64; b++)
        if (bmp->bits[b])
            allocated += __builtin_popcountl(bmp->bits[b]);
    return allocated;
}

// GCOV_EXCL_START

void utils_bitmap_print_bits(utils_bitmap_s *bmp)
{
    int b;
    int s;

    for (b = 0; b < (bmp->size_bits + 63) / 64; b++) {
        uint64_t cchunk = bmp->bits[b];
        for (s = 0; s < 64; s++) {
            fprintf(stderr, "%d", (int) (cchunk & 1));
            cchunk >>= 1;
        }
        fprintf(stderr, " ");
    }
    fprintf(stderr, "\n");
}

// GCOV_EXCL_STOP
