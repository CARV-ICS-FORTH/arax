#include "utils/bitmap.h"
#include <set>
#include <iostream>
#include "testing.h"

void alloc_bits_check(size_t size, utils_bitmap_s *bmp)
{
    size_t count = utils_bitmap_count_allocated(bmp);

    REQUIRE(size == count);
    REQUIRE(utils_bitmap_used(bmp) == count);
}

typedef size_t SingleBitAllocation;

struct MultipleBitAllocation
{
    size_t start_bit;
    size_t size;
};

bool operator < (const MultipleBitAllocation & l, const MultipleBitAllocation & r)
{
    return l.start_bit < r.start_bit;
}

std::ostream & operator << (std::ostream & os, const MultipleBitAllocation & alloc)
{
    os << alloc.start_bit << "[" << alloc.size << "]";
    return os;
}

template <class T>
void fill_vector_seq(std::vector<T> & vec, int bits, std::mt19937_64 &mersenne);

template <>
void fill_vector_seq(std::vector<SingleBitAllocation> & vec, int bits, std::mt19937_64 &mersenne)
{
    REQUIRE(vec.size() == 0);
    vec.resize(bits);
    std::iota(vec.begin(), vec.end(), 0);
}

template <>
void fill_vector_seq(std::vector<MultipleBitAllocation> & vec, int bits, std::mt19937_64 &mersenne)
{
    REQUIRE(vec.size() == 0);
    do{
        size_t size = std::min(63 + (mersenne() % 256), (size_t) bits);
        bits -= size;
        MultipleBitAllocation mba = { 0, size };
        vec.push_back(mba);
    }while(bits);
}

template <class T>
void fill_vector_stripe(std::vector<T> & vec, int bits, size_t stripe);

template <>
void fill_vector_stripe(std::vector<SingleBitAllocation> & vec, int bits, size_t stripe)
{ // SingleBitAllocation will ignore the stripe size
    std::mt19937_64 mt;

    fill_vector_seq(vec, bits, mt);
}

template <>
void fill_vector_stripe(std::vector<MultipleBitAllocation> & vec, int bits, size_t stripe)
{
    REQUIRE(vec.size() == 0);
    do{
        size_t size = std::min(stripe, (size_t) bits);
        bits -= size;
        MultipleBitAllocation mba = { 0, size };
        vec.push_back(mba);
    }while(bits);
}

template <class T>
void test_alloc(utils_bitmap_s *bmp, size_t &used_cnt, size_t bits, T & allocation, std::vector<T> & released_vector);

template <>
void test_alloc(utils_bitmap_s *bmp, size_t &used_cnt, size_t bits, SingleBitAllocation & allocation,
  std::vector<SingleBitAllocation> & used_vector)
{
    REQUIRE(utils_bitmap_used(bmp) == used_cnt);
    used_cnt++;
    size_t bit = utils_bitmap_alloc_bits(bmp, 1);

    REQUIRE(bit < bits);              // Ensure bit index is within bounds
    REQUIRE(bit != BITMAP_NOT_FOUND); // Should always have space
    allocation = bit;
    REQUIRE(utils_bitmap_used(bmp) == used_cnt); // Verify counter
    alloc_bits_check(used_cnt, bmp);             // Verify actual bitmap
    used_vector.push_back(allocation);           // Track allocation
}

template <>
void test_alloc(utils_bitmap_s *bmp, size_t &used_cnt, size_t bits, MultipleBitAllocation & allocation,
  std::vector<MultipleBitAllocation> & used_vector)
{
    REQUIRE(utils_bitmap_used(bmp) == used_cnt);
    used_cnt += allocation.size;
    size_t bit = utils_bitmap_alloc_bits(bmp, allocation.size);

    REQUIRE(bit < bits);                         // Ensure start  index is within bounds
    REQUIRE(bit + (allocation.size - 1) < bits); // Ensure end  index is within bounds
    allocation.start_bit = bit;
    REQUIRE(bit != BITMAP_NOT_FOUND);            // Should always have space
    REQUIRE(utils_bitmap_used(bmp) == used_cnt); // Verify counter
    alloc_bits_check(used_cnt, bmp);             // Verify actual bitmap
    used_vector.push_back(allocation);           // Track allocation
}

template <class T>
void test_alloc_splitable(utils_bitmap_s *bmp, size_t &used_cnt, size_t bits, T & allocation,
  std::vector<T> & used_vector);

template <>
void test_alloc_splitable(utils_bitmap_s *bmp, size_t &used_cnt, size_t bits, SingleBitAllocation & allocation,
  std::vector<SingleBitAllocation> & used_vector)
{
    test_alloc(bmp, used_cnt, bits, allocation, used_vector);
}

template <>
void test_alloc_splitable(utils_bitmap_s *bmp, size_t &used_cnt, size_t bits, MultipleBitAllocation & allocation,
  std::vector<MultipleBitAllocation> & used_vector)
{
    REQUIRE(allocation.size);
    REQUIRE(utils_bitmap_used(bmp) == used_cnt);
    size_t bit = utils_bitmap_alloc_bits(bmp, allocation.size);

    if (bit != BITMAP_NOT_FOUND) {
        used_cnt += allocation.size;
        REQUIRE(bit < bits);                         // Ensure start  index is within bounds
        REQUIRE(bit + (allocation.size - 1) < bits); // Ensure end  index is within bounds
        allocation.start_bit = bit;
        REQUIRE(bit != BITMAP_NOT_FOUND);            // Should always have space
        REQUIRE(utils_bitmap_used(bmp) == used_cnt); // Verify counter
        alloc_bits_check(used_cnt, bmp);             // Verify actual bitmap
        used_vector.push_back(allocation);           // Track allocation
    } else {                                         // Could not satisfy allocation, split it
        std::vector<SingleBitAllocation> sav;
        SingleBitAllocation sba = 0;
        allocation.size--;
        test_alloc(bmp, used_cnt, bits, sba, sav);                          // Allocate a bit
        used_vector.push_back(MultipleBitAllocation{ sba, 1 });             // Track 1bit allocation
        test_alloc_splitable(bmp, used_cnt, bits, allocation, used_vector); // Allocate remainder
    }
}

template <class T>
void test_free(utils_bitmap_s *bmp, size_t &used_cnt, T b);

template <>
void test_free(utils_bitmap_s *bmp, size_t &used_cnt, size_t b)
{
    REQUIRE(utils_bitmap_used(bmp) == used_cnt);
    used_cnt--;
    utils_bitmap_free_bits(bmp, b, 1);
    REQUIRE(utils_bitmap_used(bmp) == used_cnt); // Verify counter
    alloc_bits_check(used_cnt, bmp);             // Verify actual bitmap
}

template <>
void test_free(utils_bitmap_s *bmp, size_t &used_cnt, MultipleBitAllocation b)
{
    REQUIRE(utils_bitmap_used(bmp) == used_cnt);
    used_cnt -= b.size;
    utils_bitmap_free_bits(bmp, b.start_bit, b.size);
    REQUIRE(utils_bitmap_used(bmp) == used_cnt); // Verify counter
    alloc_bits_check(used_cnt, bmp);             // Verify actual bitmap
}

TEST_CASE("Corner Cases")
{
    DYNAMIC_SECTION("Overrun Case Small")
    {
        size_t bits     = 16;
        size_t bmp_size = UTILS_BITMAP_CALC_BYTES(bits);
        uint8_t *buff   = new uint8_t[bmp_size + 1];
        std::vector<MultipleBitAllocation> released_vector;

        memset(buff, 0xFF, bmp_size + 1);

        utils_bitmap_s *bmp = utils_bitmap_init(buff, bits);

        size_t used_cnt = 0;
        MultipleBitAllocation mba = { 0, 15 };

        test_alloc(bmp, used_cnt, bits, mba, released_vector);
        mba.start_bit = 1;
        mba.size      = 3;
        test_free(bmp, used_cnt, mba);

        /**
         * Although the bit map has 4 bits available, they are not sequential.
         * The bug was that the last chuck was fully usable (64 bits),
         * even though in this example the bitmap is only 16 bits.
         * This call should return BITMAP_NOT_FOUND.
         * It used to return bit 16, this would also use bits 17-19
         * which are out of bounds.
         */
        REQUIRE(utils_bitmap_alloc_bits(bmp, 4) == BITMAP_NOT_FOUND);
        REQUIRE(0xFF == buff[bmp_size]);

        delete buff;
    }

    DYNAMIC_SECTION("Overrun Case Large")
    { // Same kind of bug, but for 'large' allocations
        size_t bits     = 66;
        size_t bmp_size = UTILS_BITMAP_CALC_BYTES(bits);
        uint8_t *buff   = new uint8_t[bmp_size + 1];
        std::vector<MultipleBitAllocation> released_vector;

        memset(buff, 0xFF, bmp_size + 1);

        utils_bitmap_s *bmp = utils_bitmap_init(buff, bits);

        size_t used_cnt = 0;
        MultipleBitAllocation mba = { 0, 55 };

        test_alloc(bmp, used_cnt, bits, mba, released_vector);
        mba.start_bit = 10;
        mba.size      = 4;
        test_free(bmp, used_cnt, mba);

        REQUIRE(utils_bitmap_alloc_bits(bmp, 11) == BITMAP_NOT_FOUND);
        REQUIRE(0xFF == buff[bmp_size]);

        delete buff;
    }
}

TEMPLATE_TEST_CASE("bitmap_test", "", SingleBitAllocation, MultipleBitAllocation)
{
    std::mt19937_64 mersenne;

    for (size_t bits = 1; bits <= 64 * 5; bits++) {
        size_t bmp_size = UTILS_BITMAP_CALC_BYTES(bits);
        uint8_t *buff   = new uint8_t[bmp_size + 1];

        memset(buff, 0xFF, bmp_size + 1);

        DYNAMIC_SECTION("Initialize Bitmap #" << bits)
        {
            utils_bitmap_s *bmp = utils_bitmap_init(buff, bits);

            REQUIRE((void *) bmp == (void *) buff);

            REQUIRE(utils_bitmap_size(bmp) == bits);
            REQUIRE(utils_bitmap_used(bmp) == 0);
            REQUIRE(utils_bitmap_free(bmp) == bits);

            for (int pass = 1; pass < 4; pass++) {
                DYNAMIC_SECTION("Full Alloc Test #" << bits << " pass #" << pass)
                {
                    REQUIRE(utils_bitmap_alloc_bits(bmp, bits) != BITMAP_NOT_FOUND);
                    REQUIRE(utils_bitmap_used(bmp) == bits);
                    REQUIRE(utils_bitmap_free(bmp) == 0);
                    alloc_bits_check(bits, bmp);

                    // Lets try to allocate - all should fail
                    for (int size = 1; size < bits * 2; size++)
                        REQUIRE(utils_bitmap_alloc_bits(bmp, bits) == BITMAP_NOT_FOUND);

                    utils_bitmap_free_bits(bmp, 0, bits);
                    REQUIRE(utils_bitmap_used(bmp) == 0);
                    REQUIRE(utils_bitmap_free(bmp) == bits);
                    alloc_bits_check(0, bmp);
                }
            }

            DYNAMIC_SECTION("Sequential #" << bits)
            {
                std::vector<TestType> released_vector, used_vector;

                fill_vector_seq(released_vector, bits, mersenne);

                size_t used_cnt = 0;

                for (auto & b : released_vector) // Allocate all bits sequentialy
                    test_alloc(bmp, used_cnt, bits, b, used_vector);

                // Should be full
                REQUIRE(utils_bitmap_alloc_bits(bmp, 1) == BITMAP_NOT_FOUND);
                alloc_bits_check(bits, bmp);

                for (auto b : released_vector) // Free all bits sequentialy
                    test_free(bmp, used_cnt, b);

                // Should be empty
                alloc_bits_check(0, bmp);
            }

            for (size_t stripe = 1; stripe <= bits; stripe++) {
                DYNAMIC_SECTION("Striped #" << bits)
                {
                    std::vector<TestType> released_vector, used_vector;

                    fill_vector_stripe(released_vector, bits, stripe);

                    size_t used_cnt = 0;

                    for (auto & b : released_vector) // Allocate all bits sequentialy
                        test_alloc(bmp, used_cnt, bits, b, used_vector);

                    // Should be full
                    REQUIRE(utils_bitmap_alloc_bits(bmp, 1) == BITMAP_NOT_FOUND);
                    alloc_bits_check(bits, bmp);

                    for (auto b : released_vector) // Free all bits sequentialy
                        test_free(bmp, used_cnt, b);

                    // Should be empty
                    alloc_bits_check(0, bmp);
                }
            }

            DYNAMIC_SECTION("Random #" << bits)
            {
                std::vector<TestType> released_vector, used_vector;

                size_t used_cnt = 0;

                for (int pass = 0; pass < std::min(10 * bits, (size_t) 20); pass++) {
                    DYNAMIC_SECTION("Pass #" << pass)
                    {
                        fill_vector_seq(released_vector, bits, mersenne);

                        shuffle(released_vector.begin(), released_vector.end(), mersenne);

                        for (auto & b : released_vector) // Allocate all bits randomly
                            test_alloc(bmp, used_cnt, bits, b, used_vector);

                        // Should be full
                        REQUIRE(utils_bitmap_alloc_bits(bmp, 1) == BITMAP_NOT_FOUND);
                        alloc_bits_check(bits, bmp);

                        for (auto b : released_vector) // Free all bits randomly
                            test_free(bmp, used_cnt, b);

                        // Should be empty
                        alloc_bits_check(0, bmp);
                    }
                }

                DYNAMIC_SECTION("Mixed")
                {
                    fill_vector_seq(released_vector, bits, mersenne);
                    std::vector<TestType> used_vector;

                    for (int iterations = 0; iterations < 200; iterations++) {
                        bool do_free = 0;

                        if (used_vector.size() && released_vector.size() )
                            do_free = (mersenne() % 100) > 50;
                        else
                            do_free = used_vector.size();

                        shuffle(released_vector.begin(), released_vector.end(), mersenne);
                        shuffle(used_vector.begin(), used_vector.end(), mersenne);

                        if (do_free) { // Free stuff
                            TestType temp = used_vector.back();
                            used_vector.pop_back();
                            released_vector.push_back(temp);
                            test_free(bmp, used_cnt, temp);
                        } else { // Alloc stuff
                            TestType temp = released_vector.back();
                            released_vector.pop_back();
                            test_alloc_splitable(bmp, used_cnt, bits, temp, used_vector);
                        }
                    }

                    for (auto & a : used_vector)
                        test_free(bmp, used_cnt, a);

                    // Should be empty
                    alloc_bits_check(0, bmp);
                }
            }

            REQUIRE(0xFF == buff[bmp_size]);
        }
        free(buff);
    }
}
